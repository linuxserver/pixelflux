use image::{ImageBuffer, Rgba};
use std::io::Cursor as IoCursor;
use std::io::Read;
use std::time::Duration;
use xcursor::{
    parser::{parse_xcursor, Image},
    CursorTheme,
};

/// @brief Manages XCursor themes, loading, and animation state.
pub struct Cursor {
    icons: Vec<Image>,
    theme: CursorTheme,
    size: u32,
}

impl Cursor {
    /// @brief Loads the default cursor theme defined by environment variables or falls back to a generated default.
    /// @return Cursor: The initialized cursor manager.
    pub fn load() -> Cursor {
        let name = std::env::var("XCURSOR_THEME").unwrap_or_else(|_| "default".into());
        let size = std::env::var("XCURSOR_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(24);

        let theme = CursorTheme::load(&name);
        let icons = load_icon(&theme, "default").unwrap_or_else(|_| {
            let size = 16;
            let mut pixels = Vec::with_capacity((size * size * 4) as usize);
            for _ in 0..(size * size) {
                pixels.extend_from_slice(&[255, 0, 0, 255]);
            }

            vec![Image {
                size,
                width: size,
                height: size,
                xhot: 0,
                yhot: 0,
                delay: 1,
                pixels_rgba: pixels,
                pixels_argb: vec![],
            }]
        });

        Cursor { icons, theme, size }
    }

    /// @brief Retrieves the default cursor image frame for a specific timestamp and scale.
    /// @input scale: The scaling factor for the cursor size.
    /// @input time: The current duration for animation calculation.
    /// @return Image: The parsed XCursor image frame.
    pub fn get_image(&self, scale: u32, time: Duration) -> Image {
        let size = self.size * scale;
        frame(time.as_millis() as u32, size, &self.icons)
    }

    /// @brief Retrieves a named cursor image frame (e.g., "hand1") for a specific timestamp and scale.
    /// @input name: The name of the cursor icon.
    /// @input scale: The scaling factor.
    /// @input time: The current duration for animation.
    /// @return Option<Image>: The image frame if found.
    pub fn get_image_by_name(&self, name: &str, scale: u32, time: Duration) -> Option<Image> {
        let icons = load_icon(&self.theme, name).ok()?;
        let size = self.size * scale;
        Some(frame(time.as_millis() as u32, size, &icons))
    }

    /// @brief Converts a named cursor icon into a PNG byte array for transmission to web clients.
    /// @input name: The name of the cursor icon.
    /// @return Option<(Vec<u8>, u32, u32)>: Tuple containing PNG bytes, hotspot X, and hotspot Y.
    pub fn get_png_data(&self, name: &str) -> Option<(Vec<u8>, u32, u32)> {
        let icons = load_icon(&self.theme, name).ok()?;
        let image_data = nearest_images(self.size, &icons).next()?;

        let img_buf: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_raw(
            image_data.width,
            image_data.height,
            image_data.pixels_rgba.clone(),
        )?;

        let mut bytes: Vec<u8> = Vec::new();
        let mut cursor = IoCursor::new(&mut bytes);
        img_buf
            .write_to(&mut cursor, image::ImageFormat::Png)
            .ok()?;

        Some((bytes, image_data.xhot, image_data.yhot))
    }
}

/// @brief Filters available cursor images to find those matching the requested size.
/// @input size: Target size.
/// @input images: Slice of available images.
/// @return impl Iterator: Iterator over matching images.
fn nearest_images(size: u32, images: &[Image]) -> impl Iterator<Item = &Image> {
    let nearest_image = images
        .iter()
        .min_by_key(|image| (size as i32 - image.size as i32).abs())
        .unwrap();
    images
        .iter()
        .filter(move |image| image.width == nearest_image.width && image.height == nearest_image.height)
}

/// @brief Calculates the correct animation frame based on elapsed time.
/// @input millis: Elapsed milliseconds.
/// @input size: Target size.
/// @input images: Slice of available images.
/// @return Image: The specific frame to render.
fn frame(mut millis: u32, size: u32, images: &[Image]) -> Image {
    let total = nearest_images(size, images).fold(0, |acc, image| acc + image.delay);
    if total == 0 {
        return nearest_images(size, images).next().unwrap().clone();
    }
    millis %= total;
    for img in nearest_images(size, images) {
        if millis < img.delay {
            return img.clone();
        }
        millis -= img.delay;
    }
    unreachable!()
}

/// @brief Loads and parses an XCursor file from the theme.
/// @input theme: The cursor theme.
/// @input name: The icon name.
/// @return Result<Vec<Image>, String>: List of parsed images or error.
fn load_icon(theme: &CursorTheme, name: &str) -> Result<Vec<Image>, String> {
    let icon_path = theme.load_icon(name).ok_or("Icon not found")?;
    let mut cursor_file = std::fs::File::open(icon_path).map_err(|e| e.to_string())?;
    let mut cursor_data = Vec::new();
    cursor_file
        .read_to_end(&mut cursor_data)
        .map_err(|e| e.to_string())?;
    parse_xcursor(&cursor_data).ok_or("Failed to parse".to_string())
}
