use smithay::{
    backend::{
        allocator::Fourcc,
        renderer::{
            element::{
                memory::{MemoryRenderBuffer, MemoryRenderBufferRenderElement},
                Kind,
            },
            ImportMem, Renderer, Texture,
        },
    },
    utils::{Point, Transform},
};
use std::path::Path;

/// @brief Defines the screen position for the watermark overlay.
#[derive(Clone, Copy, PartialEq)]
pub enum WatermarkLocation {
    None = 0,
    TL = 1,
    TR = 2,
    BL = 3,
    BR = 4,
    MI = 5,
    AN = 6,
}

impl From<i32> for WatermarkLocation {
    fn from(v: i32) -> Self {
        match v {
            1 => Self::TL,
            2 => Self::TR,
            3 => Self::BL,
            4 => Self::BR,
            5 => Self::MI,
            6 => Self::AN,
            _ => Self::None,
        }
    }
}

/// @brief Manages the pixel data, position, and animation state of an overlay image.
pub struct OverlayState {
    wm_pixels: Vec<u8>,
    wm_width: u32,
    wm_height: u32,
    wm_pos_x: i32,
    wm_pos_y: i32,
    wm_velocity_x: f64,
    wm_velocity_y: f64,
    wm_subpixel_x: f64,
    wm_subpixel_y: f64,
    wm_loaded: bool,
    is_animated: bool,
}

impl Default for OverlayState {
    fn default() -> Self {
        Self {
            wm_pixels: Vec::new(),
            wm_width: 0,
            wm_height: 0,
            wm_pos_x: 0,
            wm_pos_y: 0,
            wm_velocity_x: 2.0,
            wm_velocity_y: 2.0,
            wm_subpixel_x: 0.0,
            wm_subpixel_y: 0.0,
            wm_loaded: false,
            is_animated: false,
        }
    }
}

impl OverlayState {
    /// @brief Loads an image from disk to use as the watermark.
    /// @input path: Filesystem path to the image.
    pub fn load_watermark(&mut self, path: &str) {
        if let Ok(img) = image::open(Path::new(path)) {
            let rgba = img.to_rgba8();
            self.wm_width = rgba.width();
            self.wm_height = rgba.height();
            self.wm_pixels = rgba.into_vec();
            self.wm_loaded = true;
        } else {
            self.wm_loaded = false;
            self.wm_pixels.clear();
        }
    }

    /// @brief Checks if a watermark is currently loaded.
    /// @return bool: True if loaded.
    pub fn is_active(&self) -> bool {
        self.wm_loaded
    }

    /// @brief Checks if the current watermark mode requires continuous animation updates.
    /// @return bool: True if animated.
    pub fn is_animated(&self) -> bool {
        self.is_animated
    }

    /// @brief Updates the watermark coordinates based on the frame size and location setting.
    /// @input frame_width: Width of the target frame.
    /// @input frame_height: Height of the target frame.
    /// @input loc_enum: Integer representation of WatermarkLocation.
    pub fn update_position(&mut self, frame_width: i32, frame_height: i32, loc_enum: i32) {
        if !self.wm_loaded {
            return;
        }

        let loc = WatermarkLocation::from(loc_enum);
        let w = self.wm_width as i32;
        let h = self.wm_height as i32;

        self.is_animated = matches!(loc, WatermarkLocation::AN);

        match loc {
            WatermarkLocation::TL => {
                self.wm_pos_x = 0;
                self.wm_pos_y = 0;
            }
            WatermarkLocation::TR => {
                self.wm_pos_x = frame_width - w;
                self.wm_pos_y = 0;
            }
            WatermarkLocation::BL => {
                self.wm_pos_x = 0;
                self.wm_pos_y = frame_height - h;
            }
            WatermarkLocation::BR => {
                self.wm_pos_x = frame_width - w;
                self.wm_pos_y = frame_height - h;
            }
            WatermarkLocation::MI => {
                self.wm_pos_x = (frame_width - w) / 2;
                self.wm_pos_y = (frame_height - h) / 2;
            }
            WatermarkLocation::AN => {
                self.wm_subpixel_x += self.wm_velocity_x;
                self.wm_subpixel_y += self.wm_velocity_y;

                if self.wm_subpixel_x <= 0.0 {
                    self.wm_subpixel_x = 0.0;
                    self.wm_velocity_x = self.wm_velocity_x.abs();
                } else if self.wm_subpixel_x + (w as f64) >= frame_width as f64 {
                    self.wm_subpixel_x = (frame_width - w) as f64;
                    self.wm_velocity_x = -self.wm_velocity_x.abs();
                }

                if self.wm_subpixel_y <= 0.0 {
                    self.wm_subpixel_y = 0.0;
                    self.wm_velocity_y = self.wm_velocity_y.abs();
                } else if self.wm_subpixel_y + (h as f64) >= frame_height as f64 {
                    self.wm_subpixel_y = (frame_height - h) as f64;
                    self.wm_velocity_y = -self.wm_velocity_y.abs();
                }

                self.wm_pos_x = self.wm_subpixel_x as i32;
                self.wm_pos_y = self.wm_subpixel_y as i32;
            }
            WatermarkLocation::None => {}
        }
    }

    /// @brief Blends the watermark pixels onto a raw CPU buffer.
    /// @input frame_buffer: Mutable slice of the frame's raw RGBA/BGRA data.
    /// @input width: Width of the frame buffer.
    /// @input height: Height of the frame buffer.
    /// @input swap_rb: If true, swaps Red and Blue channels during blending.
    pub fn apply(&self, frame_buffer: &mut [u8], width: i32, height: i32, swap_rb: bool) {
        if !self.wm_loaded {
            return;
        }

        let wm_w = self.wm_width as i32;
        let wm_h = self.wm_height as i32;

        let start_x = self.wm_pos_x.max(0);
        let start_y = self.wm_pos_y.max(0);
        let end_x = (self.wm_pos_x + wm_w).min(width);
        let end_y = (self.wm_pos_y + wm_h).min(height);

        if start_x >= end_x || start_y >= end_y {
            return;
        }

        let frame_stride = width as usize * 4;
        let wm_stride = self.wm_width as usize * 4;

        let wm_offset_x = (start_x - self.wm_pos_x) as usize;
        let wm_offset_y = (start_y - self.wm_pos_y) as usize;

        for y in 0..(end_y - start_y) {
            let frame_row_idx = ((start_y + y) as usize * frame_stride) + (start_x as usize * 4);
            let wm_row_idx = ((wm_offset_y + y as usize) * wm_stride) + (wm_offset_x * 4);

            let row_width = (end_x - start_x) as usize;

            for x in 0..row_width {
                let f_idx = frame_row_idx + (x * 4);
                let w_idx = wm_row_idx + (x * 4);

                let alpha = self.wm_pixels[w_idx + 3] as u16;
                if alpha == 0 {
                    continue;
                }

                let inv_alpha = 255 - alpha;

                let mut s_r = self.wm_pixels[w_idx] as u16;
                let s_g = self.wm_pixels[w_idx + 1] as u16;
                let mut s_b = self.wm_pixels[w_idx + 2] as u16;

                let d_r = frame_buffer[f_idx] as u16;
                let d_g = frame_buffer[f_idx + 1] as u16;
                let d_b = frame_buffer[f_idx + 2] as u16;

                if swap_rb {
                    std::mem::swap(&mut s_r, &mut s_b);
                }

                frame_buffer[f_idx] = ((s_r * alpha + d_r * inv_alpha) / 255) as u8;
                frame_buffer[f_idx + 1] = ((s_g * alpha + d_g * inv_alpha) / 255) as u8;
                frame_buffer[f_idx + 2] = ((s_b * alpha + d_b * inv_alpha) / 255) as u8;
            }
        }
    }

    /// @brief Creates a Smithay render element representing a software cursor.
    /// @input renderer: The active renderer instance.
    /// @input image: The cursor image data from xcursor.
    /// @input pos: The logical position of the cursor.
    /// @return Option<MemoryRenderBufferRenderElement>: The renderable element.
    pub fn get_cursor_element<R>(
        &self,
        renderer: &mut R,
        image: xcursor::parser::Image,
        pos: Point<i32, smithay::utils::Logical>,
    ) -> Option<MemoryRenderBufferRenderElement<R>>
    where
        R: Renderer + ImportMem,
        R::TextureId: Texture + Clone + Send + 'static,
    {
        let buffer = MemoryRenderBuffer::from_slice(
            &image.pixels_rgba,
            Fourcc::Abgr8888,
            (image.width as i32, image.height as i32),
            1,
            Transform::Normal,
            None,
        );

        let hot: Point<i32, smithay::utils::Physical> =
            (image.xhot as i32, image.yhot as i32).into();
        let phys_pos: Point<i32, smithay::utils::Physical> = (pos.x, pos.y).into();

        MemoryRenderBufferRenderElement::from_buffer(
            renderer,
            (phys_pos - hot).to_f64(),
            &buffer,
            Some(1.0),
            None,
            None,
            Kind::Cursor,
        )
        .ok()
    }
}
