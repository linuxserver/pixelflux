/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

use image::{ImageBuffer, Rgba};
use std::io::Cursor as IoCursor;
use std::io::Read;
use std::time::Duration;
use xcursor::{
    parser::{parse_xcursor, Image},
    CursorTheme,
};

/// @brief The loaded XCursor theme, held for the whole capture so cursor lookups stay cheap.
///
/// The default cursor's frames are parsed once and kept here because they are consulted on nearly
/// every frame; the theme handle is retained alongside them so the rarer named cursors (`hand1`,
/// `text`, …) can still be resolved on demand. Everything is sized to the one resolved pixel size.
pub struct Cursor {
    icons: Vec<Image>,
    theme: CursorTheme,
    size: u32,
}

impl Cursor {
    /// @brief Load the theme named by `XCURSOR_THEME` (default `"default"`) at the requested size.
    ///
    /// `size_override` comes from `CaptureSettings.cursor_size` (selkies `--cursor-size` /
    /// `XCURSOR_SIZE`); a value `<= 0` falls back to 24. When the theme's default icon cannot be
    /// loaded, a 16×16 solid-red placeholder stands in so the caller always has a valid image.
    pub fn load(size_override: i32) -> Cursor {
        let name = std::env::var("XCURSOR_THEME").unwrap_or_else(|_| "default".into());
        let size: u32 = if size_override > 0 { size_override as u32 } else { 24 };

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

    /// @brief The default cursor's animation frame for `time`, at the theme size times `scale`.
    pub fn get_image(&self, scale: u32, time: Duration) -> Image {
        let size = self.size * scale;
        frame(time.as_millis() as u32, size, &self.icons)
    }

    /// @brief A named cursor's animation frame for `time`, or `None` when the theme lacks it.
    pub fn get_image_by_name(&self, name: &str, scale: u32, time: Duration) -> Option<Image> {
        let icons = load_icon(&self.theme, name).ok()?;
        let size = self.size * scale;
        Some(frame(time.as_millis() as u32, size, &icons))
    }

    /// @brief A named cursor icon as PNG bytes plus its hotspot (x, y), for web clients.
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

/// @brief All frames of the theme variant whose pixel size is closest to `size`. XCursor files
/// bundle the same cursor at several sizes, and choosing the nearest one avoids scaling a
/// mismatched bitmap into a blurry or aliased cursor.
fn nearest_images(size: u32, images: &[Image]) -> impl Iterator<Item = &Image> {
    let nearest_image = images
        .iter()
        .min_by_key(|image| (size as i32 - image.size as i32).abs())
        .unwrap();
    images
        .iter()
        .filter(move |image| image.width == nearest_image.width && image.height == nearest_image.height)
}

/// @brief Pick which animation frame to show for the elapsed time, so animated cursors (a spinner,
/// a progress ring) actually advance instead of freezing on frame zero; it maps the time onto the
/// frames by cycling their cumulative delays.
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

/// @brief Parse the named icon from the theme's cursor file into its frames.
///
/// Empty parses are rejected so `nearest_images`'s `min_by_key().unwrap()` cannot panic on a
/// cursor file that has a valid header but zero images.
fn load_icon(theme: &CursorTheme, name: &str) -> Result<Vec<Image>, String> {
    let icon_path = theme.load_icon(name).ok_or("Icon not found")?;
    let mut cursor_file = std::fs::File::open(icon_path).map_err(|e| e.to_string())?;
    let mut cursor_data = Vec::new();
    cursor_file
        .read_to_end(&mut cursor_data)
        .map_err(|e| e.to_string())?;
    let imgs = parse_xcursor(&cursor_data).ok_or("Failed to parse".to_string())?;
    if imgs.is_empty() {
        return Err("Cursor file has no images".to_string());
    }
    Ok(imgs)
}
