import cv2
import time
import os
import numpy as np


def main():
    # Paths relative to the kornia-3d crate root
    left_path = "examples/stereo/image2_000000.png"
    right_path = "examples/stereo/image3_000000.png"

    if not os.path.exists(left_path) or not os.path.exists(right_path):
        print(f"Error: Could not find images at {left_path} and {right_path}.")
        print("Please run this script from the 'crates/kornia-3d' directory.")
        return

    # Load images in grayscale
    left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    print(f"Loaded KITTI images: {left_img.shape[1]}x{left_img.shape[0]}")

    # Map Rust parameters to OpenCV:
    # Rust: max_disparity = 128
    # Rust: census_radius = 7 => window size = 2*7 + 1 = 15
    # Rust: p1 = 10, p2 = 120
    # Rust: lr_max_diff = 1
    num_disp = 128
    block_size = 15

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=10,  # Raw penalty to match Rust implementation
        P2=120,  # Raw penalty to match Rust implementation
        disp12MaxDiff=1,
        uniquenessRatio=0,
        speckleWindowSize=0,
        speckleRange=0,
        mode=cv2.STEREO_SGBM_MODE_HH,
    )

    # Warmup run (avoids counting initialization overhead)
    stereo.compute(left_img, right_img)

    # Benchmark
    iterations = 5
    start = time.time()
    for _ in range(iterations):
        disparity_raw = stereo.compute(left_img, right_img)
    end = time.time()

    avg_time_ms = (end - start) / iterations * 1000.0
    print(f"\nOpenCV StereoSGBM Time: {avg_time_ms:.2f} ms per frame")

    # ==========================================
    # Post-processing and Saving the Disparity Map
    # ==========================================

    # 1. Convert from CV_16S (fixed point) to float32
    # OpenCV stores disparity * 16. We divide by 16.0 to get true pixel disparity.
    disp_f32 = disparity_raw.astype(np.float32) / 16.0

    # 2. Create a mask for invalid pixels.
    # OpenCV sets invalid pixels to (minDisparity - 1), which is -1 in CV_16S.
    # In float32, this is -0.0625. So anything <= 0 is invalid (equivalent to NaN in Rust).
    invalid_mask = disp_f32 <= 0.0

    # 3. Normalize valid disparities to 0-255 for visualization
    disp_valid = np.where(invalid_mask, 0, disp_f32)
    disp_norm = cv2.normalize(
        disp_valid, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # 4. Save Grayscale Version
    cv2.imwrite("opencv_disparity_gray.png", disp_norm)
    print("Saved: opencv_disparity_gray.png")

    # 5. Save Colorized Version (matching the Rust turbo-ish colormap)
    # Apply OpenCV's TURBO colormap
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

    # Force invalid pixels to be pure black [0, 0, 0], exactly like the Rust script
    disp_color[invalid_mask] = [0, 0, 0]

    cv2.imwrite("opencv_disparity_color.png", disp_color)
    print("Saved: opencv_disparity_color.png")


if __name__ == "__main__":
    main()
