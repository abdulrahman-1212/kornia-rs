//! Semi-Global Matching (SGM) stereo disparity estimation.
//!
//! Given a rectified stereo pair (output of [`StereoRectifier`]), this module
//! computes a dense disparity map via SGM.
//!
//! ## Pipeline
//!
//! 1. **Census transform** - 5x5 bit-string descriptor around each pixel;
//!    robust to gain/bias differences between the two views.
//! 2. **Matching cost volume** - Hamming distance between left and
//!    right census strings for each `(u, v, d)` triplet, `d belongs to [0, max_disp)`.
//! 3. **SGM aggregation** - path costs summed over 8 directions with the
//!    standard smoothness penalties P1 (small steps) and P2 (large steps).
//! 4. **Winner-takes-all (WTA)** - per-pixel minimum of the aggregated cost.
//! 5. **Sub-pixel refinement** - parabolic fit over the three costs around
//!    the WTA minimum; improves accuracy to ~0.5 px.
//! 6. **Left-right consistency check** - pixels whose left and right WTA
//!    minima disagree by more than `lr_max_diff` are marked invalid (`NaN`).
//!
//! Mirrors the structure of OpenCV's `StereoSGBM`, but written from scratch
//! in safe Rust with no external C dependencies.

use kornia_image::Image;

/// Errors produced by [`StereoMatcher`].
#[derive(Debug, thiserror::Error)]
pub enum MatcherError {
    /// The supplied images have different sizes.
    #[error("left image {left:?} and right image {right:?} must have the same size")]
    SizeMismatch {
        /// Left image `(width, height)`.
        left: (usize, usize),
        /// Right image `(width, height)`.
        right: (usize, usize),
    },
    /// `max_disparity` must be > 0 and a multiple of the block size.
    #[error("max_disparity must be > 0, got {0}")]
    InvalidMaxDisparity(usize),
    /// P1 / P2 penalties are nonsensical.
    #[error("SGM penalties must satisfy 0 < P1 < P2, got P1={p1} P2={p2}")]
    InvalidPenalties {
        /// P1 value.
        p1: u16,
        /// P2 value.
        p2: u16,
    },
    /// Failed to allocate the output image.
    #[error(transparent)]
    Image(#[from] kornia_image::ImageError),
}

/// A disparity map: one `f32` per pixel, `NaN` where invalid.
///
/// Disparity is in pixels of the *rectified* image pair.
pub struct DisparityMap {
    /// Width of the map in pixels.
    pub width: usize,
    /// Height of the map in pixels.
    pub height: usize,
    /// Row-major disparity values. `NaN` marks occluded / invalid pixels.
    pub data: Vec<f32>,
}

impl DisparityMap {
    /// Returns the disparity at pixel `(u, v)`, or `None` if out-of-bounds.
    pub fn get(&self, u: usize, v: usize) -> Option<f32> {
        if u < self.width && v < self.height {
            Some(self.data[v * self.width + u])
        } else {
            None
        }
    }

    /// Total number of pixels with a valid (non-NaN) disparity.
    pub fn valid_count(&self) -> usize {
        self.data.iter().filter(|d| !d.is_nan()).count()
    }

    /// Fraction of pixels with a valid disparity, in `[0, 1]`.
    pub fn valid_fraction(&self) -> f64 {
        self.valid_count() as f64 / (self.width * self.height) as f64
    }
}

/// SGM parameters.
///
/// # Defaults
///
/// The defaults are a reasonable starting point for a 640x480 stereo pair
/// with a ~0.1 m baseline and scenes up to ~5 m:
///
/// ```
/// # use kornia_3d::stereo::sgm::StereoMatcher;
/// let m = StereoMatcher::default();
/// ```
#[derive(Debug, Clone)]
pub struct StereoMatcher {
    /// Maximum disparity (exclusive). Must be > 0.
    ///
    /// A scene depth of `d_min` metres requires at least `bf / d_min` pixels
    /// of disparity range.  For EuRoC at 480p with bf = 50 px.m, 64 covers
    /// depths down to ~0.8 m.
    pub max_disparity: usize,
    /// Penalty for a disparity change of exactly 1 px between neighbours.
    ///
    /// Larger: smoother disparity transitions; typical range 5-20.
    pub p1: u16,
    /// Penalty for a disparity change > 1 px between neighbours.
    ///
    /// Must be > `p1`; typical range 50-150.  A ratio `p2 / p1 = 8` works
    /// well for most scenes.
    pub p2: u16,
    /// Half-size of the Census transform window.
    ///
    /// The full window is `(2*r+1) x (2*r+1)`.  `r=2` (5x5) fits in a
    /// `u32` and handles most textures; increase to 3 (7x7, needs `u64`)
    /// only if you observe matching errors on smooth surfaces.
    pub census_radius: usize,
    /// Left-right consistency threshold in pixels.
    ///
    /// Pixels where `|disp_L - disp_R| > lr_max_diff` are marked invalid.
    /// Set to `usize::MAX` to disable the LR check.
    pub lr_max_diff: usize,
}

impl Default for StereoMatcher {
    fn default() -> Self {
        Self {
            max_disparity: 64,
            p1: 10,
            p2: 120,
            census_radius: 3, // 5x5 window
            lr_max_diff: 1,
        }
    }
}

impl StereoMatcher {
    /// Creates a new matcher with explicit parameters.
    ///
    /// # Errors
    /// - [`MatcherError::InvalidMaxDisparity`] if `max_disparity` is 0.
    /// - [`MatcherError::InvalidPenalties`] if `p2 <= p1`.
    pub fn new(
        max_disparity: usize,
        p1: u16,
        p2: u16,
        census_radius: usize,
        lr_max_diff: usize,
    ) -> Result<Self, MatcherError> {
        if max_disparity == 0 {
            return Err(MatcherError::InvalidMaxDisparity(max_disparity));
        }
        if p2 <= p1 {
            return Err(MatcherError::InvalidPenalties { p1, p2 });
        }
        Ok(Self {
            max_disparity,
            p1,
            p2,
            census_radius,
            lr_max_diff,
        })
    }

    /// Computes a dense disparity map from a rectified stereo pair.
    ///
    /// `left` and `right` must be the outputs of [`StereoRectifier::rectify_left`]
    /// and [`StereoRectifier::rectify_right`] respectively - i.e. grayscale,
    /// same size, epipolar lines horizontal.
    ///
    /// Returns a [`DisparityMap`] with sub-pixel precision.  Invalid pixels
    /// (occlusions, LR-check failures, border regions) carry `NaN`.
    ///
    /// # Errors
    /// [`MatcherError::SizeMismatch`] if the images differ in size.
    pub fn compute_disparity(
        &self,
        left: &Image<u8, 1>,
        right: &Image<u8, 1>,
    ) -> Result<DisparityMap, MatcherError> {
        let (w, h) = (left.width(), left.height());
        if (right.width(), right.height()) != (w, h) {
            return Err(MatcherError::SizeMismatch {
                left: (w, h),
                right: (right.width(), right.height()),
            });
        }

        let left_src = left.as_slice();
        let right_src = right.as_slice();
        let d = self.max_disparity;

        // 1. Census transform.
        let left_census = census_transform(left_src, w, h, self.census_radius);
        let right_census = census_transform(right_src, w, h, self.census_radius);

        // 2. Matching cost volume C[v][u][d] = Hamming(left[v,u], right[v,u-d]).
        //    Stored row-major: index = (v*w + u)*d + disp.
        let cost_vol = build_cost_volume(&left_census, &right_census, w, h, d);

        // 3. SGM aggregation over 8 directions.
        let aggregated = sgm_aggregate(&cost_vol, w, h, d, self.p1, self.p2);

        // 4. WTA + sub-pixel refinement → left disparity map.
        let disp_left = wta_subpixel(&aggregated, w, h, d);

        // 5. Compute right disparity for LR consistency check.
        //    Right cost volume: C_R[v][u][d] = Hamming(right[v,u], left[v,u+d]).
        let cost_vol_r = build_cost_volume_right(&left_census, &right_census, w, h, d);
        let aggregated_r = sgm_aggregate(&cost_vol_r, w, h, d, self.p1, self.p2);
        let disp_right = wta_subpixel(&aggregated_r, w, h, d);

        // 6. LR consistency check.
        let data = lr_check(&disp_left, &disp_right, w, h, self.lr_max_diff);

        Ok(DisparityMap {
            width: w,
            height: h,
            data,
        })
    }
}

// ---------------------------------------------------------------------------
// Census transform
// ---------------------------------------------------------------------------

/// Computes a Census bit-string for every pixel using a `(2r+1)x(2r+1)` window.
///
/// Pixel at `(u, v)` is compared against its `(2r+1)^2-1` neighbours; each
/// comparison contributes one bit (1 if neighbour < centre).  The result is
/// packed into a `u64`; for `r=2` (5x5 = 24 bits) this fits comfortably.
/// Pixels within `r` of the border receive a value of `0`.
fn census_transform(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u64> {
    let mut out = vec![0u64; w * h];
    let r_i = r as isize;
    for v in r..h - r {
        for u in r..w - r {
            let centre = src[v * w + u];
            let mut bits = 0u64;
            let mut bit_idx = 0u32;
            for dv in -r_i..=r_i {
                for du in -r_i..=r_i {
                    if du == 0 && dv == 0 {
                        continue;
                    }
                    let nu = (u as isize + du) as usize;
                    let nv = (v as isize + dv) as usize;
                    if src[nv * w + nu] < centre {
                        bits |= 1u64 << bit_idx;
                    }
                    bit_idx += 1;
                }
            }
            out[v * w + u] = bits;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Cost volume
// ---------------------------------------------------------------------------

/// Builds a left-reference cost volume.
///
/// `C[v, u, d] = hamming(left_census[v,u], right_census[v, u-d])`.
/// Right pixels out of bounds default to cost 64 (maximum Hamming distance
/// for a 64-bit descriptor), which penalises invalid matches.
fn build_cost_volume(left: &[u64], right: &[u64], w: usize, h: usize, max_d: usize) -> Vec<u16> {
    let mut vol = vec![u16::MAX; h * w * max_d];
    for v in 0..h {
        let row = v * w;
        for u in 0..w {
            for d in 0..max_d.min(u + 1) {
                let cost = (left[row + u] ^ right[row + u - d]).count_ones() as u16;
                vol[(row + u) * max_d + d] = cost;
            }
        }
    }
    vol
}

/// Builds a right-reference cost volume for the LR consistency check.
///
/// `C_R[v, u, d] = hamming(right_census[v,u], left_census[v, u+d])`.
fn build_cost_volume_right(
    left: &[u64],
    right: &[u64],
    w: usize,
    h: usize,
    max_d: usize,
) -> Vec<u16> {
    let mut vol = vec![u16::MAX; h * w * max_d];
    for v in 0..h {
        let row = v * w;
        for u in 0..w {
            for d in 0..max_d {
                let ur = u + d;
                if ur >= w {
                    break;
                }
                let cost = (right[row + u] ^ left[row + ur]).count_ones() as u16;
                vol[(row + u) * max_d + d] = cost;
            }
        }
    }
    vol
}

// ---------------------------------------------------------------------------
// SGM aggregation
// ---------------------------------------------------------------------------

/// The eight cardinal + diagonal scan directions.
///
/// Each entry is `(du, dv)` - the step taken when *scanning forward* along
/// the path. We process each direction independently in a single pass and
/// accumulate.
const DIRECTIONS: [(isize, isize); 8] = [
    (1, 0),   //
    (-1, 0),  //
    (0, 1),   //
    (0, -1),  //
    (1, 1),   //
    (-1, -1), //
    (1, -1),  //
    (-1, 1),  //
];

/// Aggregates the cost volume along all 8 directions using the SGM recurrence.
///
/// The output volume has the same layout as the input but holds aggregated
/// `u16` costs (saturating arithmetic prevents overflow).
fn sgm_aggregate(cost: &[u16], w: usize, h: usize, max_d: usize, p1: u16, p2: u16) -> Vec<u32> {
    let n = h * w * max_d;
    let mut agg = vec![0u32; n];

    // Two small reusable row buffers (length `max_d`), swapped each step.
    // Using owned rows instead of slicing into one big buffer avoids holding
    // an immutable borrow across the loop while writing a fresh row.
    let mut prev_row = vec![0u16; max_d];
    let mut cur_row = vec![0u16; max_d];

    for &(du, dv) in &DIRECTIONS {
        aggregate_direction(
            cost,
            &mut agg,
            &mut prev_row,
            &mut cur_row,
            w,
            h,
            max_d,
            du,
            dv,
            p1,
            p2,
        );
    }

    agg
}

/// Aggregates one directional path into `agg`.
///
/// For each scanline along `(du, dv)` we apply the SGM recurrence:
/// ```text
/// Lr(p, d) = C(p, d)
///          + min(Lr(p-r, d),
///                Lr(p-r, d+-1) + P1,
///                min_k Lr(p-r, k) + P2)
///          - min_k Lr(p-r, k)
/// ```
/// `prev_row` / `cur_row` are scratch buffers of length `max_d`, reused (and
/// swapped) across every scanline and every pixel to avoid per-step
/// allocation.
#[allow(clippy::too_many_arguments)]
fn aggregate_direction(
    cost: &[u16],
    agg: &mut [u32],
    prev_row: &mut Vec<u16>,
    cur_row: &mut Vec<u16>,
    w: usize,
    h: usize,
    max_d: usize,
    du: isize,
    dv: isize,
    p1: u16,
    p2: u16,
) {
    // Enumerate all scanline start pixels (those on the entry border of the
    // image for this direction).
    let starts: Vec<(usize, usize)> = entry_pixels(w, h, du, dv);

    for (u0, v0) in starts {
        let mut u = u0 as isize;
        let mut v = v0 as isize;
        let mut is_first = true;

        while u >= 0 && v >= 0 && (u as usize) < w && (v as usize) < h {
            let uu = u as usize;
            let vv = v as usize;
            let base = (vv * w + uu) * max_d;
            let cur_cost = &cost[base..base + max_d];

            if is_first {
                // First pixel on the path: cost only.
                cur_row.copy_from_slice(cur_cost);
                is_first = false;
            } else {
                let min_prev = prev_row.iter().copied().min().unwrap_or(u16::MAX);
                for d in 0..max_d {
                    let c = cur_cost[d];
                    let l0 = prev_row[d];
                    let l1 = if d > 0 {
                        prev_row[d - 1].saturating_add(p1)
                    } else {
                        u16::MAX
                    };
                    let l2 = if d + 1 < max_d {
                        prev_row[d + 1].saturating_add(p1)
                    } else {
                        u16::MAX
                    };
                    let l3 = min_prev.saturating_add(p2);
                    let best = l0.min(l1).min(l2).min(l3);
                    cur_row[d] = c.saturating_add(best).saturating_sub(min_prev);
                }
            }

            // Accumulate into the global aggregation volume.
            for d in 0..max_d {
                agg[base + d] = agg[base + d].saturating_add(cur_row[d] as u32);
            }

            // `cur_row` becomes `prev_row` for the next step; swap avoids a copy.
            std::mem::swap(prev_row, cur_row);

            u += du;
            v += dv;
        }
    }
}

/// Returns the set of pixels on the entry border for a scan direction `(du, dv)`.
///
/// Entry border = pixels from which a walk in direction `(du, dv)` covers
/// the most new ground (i.e. the border *opposite* to the direction's travel).
fn entry_pixels(w: usize, h: usize, du: isize, dv: isize) -> Vec<(usize, usize)> {
    let mut starts = Vec::new();
    // Pixels on the left border if moving right, right border if moving left.
    // Pixels on the top border if moving down, bottom border if moving up.
    // Diagonal directions start from one full edge.
    let u_start: Vec<usize> = if du > 0 {
        vec![0]
    } else if du < 0 {
        vec![w - 1]
    } else {
        (0..w).collect()
    };
    let v_start: Vec<usize> = if dv > 0 {
        vec![0]
    } else if dv < 0 {
        vec![h - 1]
    } else {
        (0..h).collect()
    };
    for &u in &u_start {
        for &v in &v_start {
            starts.push((u, v));
        }
    }
    starts
}

// ---------------------------------------------------------------------------
// WTA + sub-pixel refinement
// ---------------------------------------------------------------------------

/// Finds the winner-takes-all disparity with parabolic sub-pixel refinement.
///
/// For each pixel the disparity `d*` minimises the aggregated cost.  If the
/// minimum is not at the border of the search range, a parabolic fit over
/// `(d*-1, d*, d*+1)` refines the estimate.  Border pixels (within
/// `census_radius` of the edge, determined implicitly by cost == u16::MAX)
/// are left as `NaN`.
fn wta_subpixel(agg: &[u32], w: usize, h: usize, max_d: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; w * h];
    for v in 0..h {
        for u in 0..w {
            let base = (v * w + u) * max_d;
            let slice = &agg[base..base + max_d];

            // WTA.
            let (d_best, &c_best) = match slice.iter().enumerate().min_by_key(|&(_, &c)| c) {
                Some(val) => val,
                None => continue,
            };

            // Threshold for invalid pixels. Max Hamming distance for 24-bit census is 24.
            // With 8 directions, max aggregated cost for valid pixels ≈ 8 × 24 = 192.
            // Invalid pixels (original cost = u16::MAX) will have much higher aggregated costs.
            if c_best > 1000 {
                continue; // border / invalid
            }

            // Parabolic sub-pixel fit.
            let d_sub = if d_best > 0 && d_best + 1 < max_d {
                let c0 = slice[d_best - 1] as f32;
                let c1 = c_best as f32;
                let c2 = slice[d_best + 1] as f32;
                let denom = c0 - 2.0 * c1 + c2;
                if denom.abs() > 1e-6 {
                    d_best as f32 + 0.5 * (c0 - c2) / denom
                } else {
                    d_best as f32
                }
            } else {
                d_best as f32
            };

            out[v * w + u] = d_sub;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Left-right consistency check
// ---------------------------------------------------------------------------

/// Invalidates pixels where left and right WTA disparities are inconsistent.
///
/// At each left pixel `(u, v)` with disparity `dL`, the corresponding right
/// pixel is `(u - round(dL), v)`.  If its right disparity `dR` satisfies
/// `|dL - dR| > threshold`, the left pixel is set to `NaN`.
fn lr_check(disp_l: &[f32], disp_r: &[f32], w: usize, h: usize, threshold: usize) -> Vec<f32> {
    let thr = threshold as f32;
    let mut out = disp_l.to_vec();
    for v in 0..h {
        for u in 0..w {
            let dl = disp_l[v * w + u];
            if dl.is_nan() {
                continue;
            }
            let ur = u as isize - dl.round() as isize;
            if ur < 0 || ur as usize >= w {
                out[v * w + u] = f32::NAN;
                continue;
            }
            let dr = disp_r[v * w + ur as usize];
            if dr.is_nan() || (dl - dr).abs() > thr {
                out[v * w + u] = f32::NAN;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    fn grey_image(width: usize, height: usize, data: Vec<u8>) -> Image<u8, 1> {
        Image::from_size_slice(ImageSize { width, height }, &data).unwrap()
    }

    /// Solid grey pair — no texture, no reliable match, but the pipeline must
    /// not panic and must return some valid or NaN pixels.
    #[test]
    fn solid_grey_does_not_panic() {
        let m = StereoMatcher::default();
        let img = grey_image(64, 48, vec![128u8; 64 * 48]);
        let disp = m.compute_disparity(&img, &img).unwrap();
        assert_eq!(disp.data.len(), 64 * 48);
    }

    /// Size mismatch triggers the right error.
    #[test]
    fn size_mismatch_errors() {
        let m = StereoMatcher::default();
        let left = grey_image(64, 48, vec![0u8; 64 * 48]);
        let right = grey_image(32, 48, vec![0u8; 32 * 48]);
        assert!(matches!(
            m.compute_disparity(&left, &right),
            Err(MatcherError::SizeMismatch { .. })
        ));
    }

    /// Invalid penalty order errors early.
    #[test]
    fn invalid_penalties_error() {
        assert!(matches!(
            StereoMatcher::new(64, 100, 50, 2, 1),
            Err(MatcherError::InvalidPenalties { .. })
        ));
    }

    /// Zero max_disparity errors early.
    #[test]
    fn zero_disparity_errors() {
        assert!(matches!(
            StereoMatcher::new(0, 10, 80, 2, 1),
            Err(MatcherError::InvalidMaxDisparity(0))
        ));
    }

    /// A synthetic horizontal-shift pair: right image = left shifted by K px.
    /// After SGM the dominant disparity should be K (±1 for subpixel / border).
    #[test]
    fn horizontal_shift_recovers_disparity() {
        let (w, h) = (128usize, 64usize);
        let shift = 8usize; // known disparity
                            // Textured left image: vertical stripes of different intensities.
        let left_data: Vec<u8> = (0..w * h)
            .map(|i| {
                let u = i % w;
                // Four bands of width 8 with different grey levels.
                match (u / 8) % 4 {
                    0 => 60,
                    1 => 130,
                    2 => 200,
                    _ => 90,
                }
            })
            .collect();
        // Right image = left image shifted left by `shift` pixels (disparity = shift).
        let right_data: Vec<u8> = (0..w * h)
            .map(|i| {
                let u = i % w;
                let v = i / w;
                if u + shift < w {
                    left_data[v * w + u + shift]
                } else {
                    128
                }
            })
            .collect();

        let left = grey_image(w, h, left_data);
        let right = grey_image(w, h, right_data);

        let matcher = StereoMatcher {
            max_disparity: 32,
            p1: 8,
            p2: 64,
            census_radius: 3,
            lr_max_diff: 2,
        };

        let disp = matcher.compute_disparity(&left, &right).unwrap();

        // At least 40 % of pixels should report the correct disparity
        // (borders and occluded regions may not).
        let valid_and_correct = disp
            .data
            .iter()
            .filter(|&&d| !d.is_nan() && (d - shift as f32).abs() < 2.0)
            .count();
        let total = w * h;
        let frac = valid_and_correct as f64 / total as f64;
        assert!(
            frac > 0.40,
            "expected >40% correct disparity pixels, got {:.1}%",
            frac * 100.0
        );
    }

    /// `valid_fraction` and `valid_count` are consistent.
    #[test]
    fn valid_count_fraction_consistent() {
        let disp = DisparityMap {
            width: 4,
            height: 1,
            data: vec![1.0, f32::NAN, 2.0, f32::NAN],
        };
        assert_eq!(disp.valid_count(), 2);
        assert!((disp.valid_fraction() - 0.5).abs() < 1e-6);
    }
}
