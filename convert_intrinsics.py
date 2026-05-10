import argparse
import os
import pickle

import numpy as np

DEFAULT_CALIBRATION = 'output/calibration_data.pkl'

def load_calibration(path):
    """Load calibration data from pickle file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'Calibration file not found at {path}')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_calibration(data, path):
    """Save calibration data to pickle file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved converted calibration to {path}')

def convert_intrinsics(K_old, dim_calib, dim_video):
    """
    Converts camera intrinsics from Calibration resolution to Video resolution.
    Handles scaling (resolution change) and cropping (aspect ratio change).
    
    Args:
        K_old: The 3x3 camera matrix from calibration
        dim_calib: (width, height) of the images used for calibration
        dim_video: (width, height) of your actual video footage
    
    Returns:
        K_new: The converted 3x3 camera matrix
    """
    # NOTE: The Distortion Coefficients (D) do NOT change. 
    # You use the exact same D values from your calibration.
    w_calib, h_calib = dim_calib
    w_video, h_video = dim_video

    # 1. Calculate the Scale Factor
    # Calculate how much each dimension needs to change
    scale_w = w_video / w_calib
    scale_h = h_video / h_calib
    
    # Always use the SMALLER scale factor - this ensures the image fits within the target
    # and we then crop or pad the other dimension to reach the target size
    scale = min(scale_w, scale_h)

    # 2. Scale the Intrinsics
    # Focal lengths and principal points scale linearly with resolution.
    K_new = K_old.copy()
    K_new *= scale # Scales fx, fy, cx, cy
    K_new[2, 2] = 1.0 # Restore the bottom-right 1.0

    # 3. Adjust for Crops
    # Calculate what the dimensions would be after scaling
    w_calib_scaled = w_calib * scale
    h_calib_scaled = h_calib * scale
    
    # Calculate crop amounts (handles center crop and letterbox)
    # In the latter scenario, crop will be negative
    h_crop = h_calib_scaled - h_video
    w_crop = w_calib_scaled - w_video
    
    # Adjust principal point for crops 
    # center crop will cause a shift to the left/up equal to half the crop amount
    # letterboxing will cause a shift to the right/down equal to the width of one band
    x_shift = w_crop / 2.0
    y_shift = h_crop / 2.0
    
    K_new[0, 2] -= x_shift  # cx adjustment for horizontal crop
    K_new[1, 2] -= y_shift  # cy adjustment for vertical crop

    print(f"--- Conversion Report ---")
    print(f"Resolution: {dim_calib} -> {dim_video}")
    print(f"Scale Factor: {scale:.4f}")
    if w_crop > 0:
        print(f"Horizontal: CROP {w_crop:.1f} pixels (scaled)")
    elif w_crop < 0:
        print(f"Horizontal: PAD {-w_crop:.1f} pixels (letterbox)")
    else:
        print(f"Horizontal: No crop/pad")
    if h_crop > 0:
        print(f"Vertical: CROP {h_crop:.1f} pixels (scaled)")
    elif h_crop < 0:
        print(f"Vertical: PAD {-h_crop:.1f} pixels (pillarbox)")
    else:
        print(f"Vertical: No crop/pad")
    print(f"New Matrix:\n{K_new}")
    
    return K_new

def main():
    parser = argparse.ArgumentParser(
        description='Convert camera intrinsics from one resolution to another.'
    )
    parser.add_argument(
        '-c', '--calibration-file', 
        default=DEFAULT_CALIBRATION,
        help='Path to input calibration pickle file.'
    )
    parser.add_argument(
        '-o', '--output-file',
        help='Path to save converted calibration pickle file. If not provided, prints to console only.'
    )
    parser.add_argument(
        '-cw', '--calib-width', type=int, required=True,
        help='Width of images used for calibration (pixels).'
    )
    parser.add_argument(
        '-ch', '--calib-height', type=int, required=True,
        help='Height of images used for calibration (pixels).'
    )
    parser.add_argument(
        '-vw', '--video-width', type=int, required=True,
        help='Width of target video/images (pixels).'
    )
    parser.add_argument(
        '-vh', '--video-height', type=int, required=True,
        help='Height of target video/images (pixels).'
    )
    args = parser.parse_args()

    # Load calibration data
    try:
        data = load_calibration(args.calibration_file)
    except FileNotFoundError as exc:
        print(f'Error: {exc}')
        return

    if 'camera_matrix' not in data:
        print('Error: Calibration file does not contain camera_matrix')
        return

    # Convert intrinsics
    K_old = data['camera_matrix']
    dim_calib = (args.calib_width, args.calib_height)
    dim_video = (args.video_width, args.video_height)

    K_new = convert_intrinsics(K_old, dim_calib, dim_video)

    # Update calibration data with converted matrix and target image size
    data['camera_matrix'] = K_new
    data['image_size'] = dim_video
    
    # Save if output file is specified
    if args.output_file:
        save_calibration(data, args.output_file)
    else:
        print('\nNo output file specified. Converted data not saved.')

if __name__ == '__main__':
    main()

