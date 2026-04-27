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
    utils::{Point, Transform, Physical},
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
    render_buffer: Option<MemoryRenderBuffer>,
}

impl Default for OverlayState {
    fn default() -> Self {
        Self {
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
            render_buffer: None,
        }
    }
}

impl OverlayState {
    /// @brief Loads an image from disk to use as the watermark.
    /// @input path: Filesystem path to the image.
    /// @input output_scale: The current output's fractional scale factor.
    /// Used as the buffer's logical-to-physical scale so the rendered overlay
    /// occupies the same physical extent as its source pixel dimensions.
    /// Without this, on outputs with scale > 1 the buffer is rendered at
    /// `pixel_size * output_scale` while position math (`frame_w - wm_w`)
    /// still uses `wm_w` as if it were physical, causing the watermark to
    /// extend past the right/bottom of the frame for any anchor other than
    /// TL (where both anchor coords are 0 and the scale is a no-op).
    pub fn load_watermark(&mut self, path: &str, output_scale: f64) {
        if let Ok(img) = image::open(Path::new(path)) {
            let rgba = img.to_rgba8();
            self.wm_width = rgba.width();
            self.wm_height = rgba.height();
            self.wm_loaded = true;

            // Round up so a 1.5 fractional scale doesn't clip off-screen
            // (it may inset the watermark slightly from the edge instead).
            // For integer scales (1, 2) this is exact and the math works out.
            let buffer_scale = output_scale.ceil().max(1.0) as i32;

            self.render_buffer = Some(MemoryRenderBuffer::from_slice(
                &rgba.into_vec(),
                Fourcc::Abgr8888,
                (self.wm_width as i32, self.wm_height as i32),
                buffer_scale,
                Transform::Normal,
                None,
            ));
        } else {
            self.wm_loaded = false;
            self.render_buffer = None;
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

    /// @brief Creates a Smithay render element representing the watermark.
    /// @input renderer: The active renderer instance.
    /// @return Option<MemoryRenderBufferRenderElement>: The renderable element.
    pub fn get_watermark_element<R>(
        &self,
        renderer: &mut R,
    ) -> Option<MemoryRenderBufferRenderElement<R>>
    where
        R: Renderer + ImportMem,
        R::TextureId: Texture + Clone + Send + 'static,
    {
        if let Some(buffer) = &self.render_buffer {
            let location = Point::<f64, Physical>::from((self.wm_pos_x as f64, self.wm_pos_y as f64));
            MemoryRenderBufferRenderElement::from_buffer(
                renderer,
                location,
                buffer,
                Some(1.0),
                None,
                None,
                Kind::Unspecified,
            )
            .ok()
        } else {
            None
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
