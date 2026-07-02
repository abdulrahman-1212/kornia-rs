//! Example: run SGM disparity estimation on a real KITTI stereo pair.
//!
//! ```bash
//! # from crates/kornia-3d/
//! cargo run --example sgm_kitti --release -- src/stereo/image2_000000.png src/stereo/image3_000000.png
//! ```
//!
//! Writes `disparity.png` (colorized) to the current directory.
//!
//! NOTE: KITTI's `image_2` (left) / `image_3` (right) pairs are already
//! rectified by the dataset itself - no [`StereoRectifier`] step is needed
//! here, we go straight from PNG to [`StereoMatcher`].

use image::{GenericImageView, Rgb, RgbImage};
use kornia_3d::stereo::sgm::StereoMatcher;
use kornia_image::{Image, ImageSize};

fn load_grayscale(path: &str) -> Image<u8, 1> {
    let img = image::open(path).unwrap_or_else(|e| panic!("failed to open {path}: {e}"));
    let (w, h) = img.dimensions();
    let gray = img.to_luma8();
    Image::from_size_slice(
        ImageSize {
            width: w as usize,
            height: h as usize,
        },
        gray.as_raw(),
    )
    .expect("image size mismatch")
}

/// Simple blue -> green -> yellow -> red ramp (turbo-ish) for `t in [0, 1]`.
fn colormap(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let r = (255.0 * (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0)) as u8;
    let g = (255.0 * (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0)) as u8;
    let b = (255.0 * (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0)) as u8;
    [r, g, b]
}

/// Saves a colorized visualization of a `f32` map (`NaN` → black).
fn save_colorized(width: usize, height: usize, data: &[f32], path: &str) {
    let (lo, hi) = data
        .iter()
        .filter(|v| v.is_finite())
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let range = (hi - lo).max(1e-6);

    let mut out = RgbImage::new(width as u32, height as u32);
    for (i, &v) in data.iter().enumerate() {
        let (x, y) = ((i % width) as u32, (i / width) as u32);
        let px = if v.is_finite() {
            let t = (v - lo) / range;
            colormap(t)
        } else {
            [0, 0, 0] // invalid pixel: black
        };
        out.put_pixel(x, y, Rgb(px));
    }
    out.save(path)
        .unwrap_or_else(|e| panic!("failed to save {path}: {e}"));
    println!("wrote {path}  (range [{lo:.2}, {hi:.2}])");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let left_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("src/stereo/image2_000000.png");
    let right_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("src/stereo/image3_000000.png");

    println!("loading {left_path} / {right_path}");
    let left = load_grayscale(left_path);
    let right = load_grayscale(right_path);
    println!("image size: {}x{}", left.width(), left.height());

    // KITTI driving scenes are wide-baseline / long-range; 128 px covers
    // objects down to ~2.8 m while staying reasonably fast.
    // Increase if you see clipped (saturated) disparity near close objects.
    let matcher = StereoMatcher {
        max_disparity: 128,
        p1: 10,
        p2: 120,
        census_radius: 7,
        lr_max_diff: 1,
    };

    println!("running SGM (use --release, debug builds are very slow)...");
    let disp = matcher
        .compute_disparity(&left, &right)
        .expect("SGM failed");

    println!(
        "disparity: {:.1}% valid pixels",
        disp.valid_fraction() * 100.0
    );

    save_colorized(disp.width, disp.height, &disp.data, "disparity.png");
}
