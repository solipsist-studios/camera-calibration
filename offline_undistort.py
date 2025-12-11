import argparse
import glob
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_FILE = 'output/calibration_data.pkl'
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def load_calibration(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Calibration file not found at {path}')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    mtx = data.get('camera_matrix')
    dist = data.get('distortion_coefficients')
    if mtx is None or dist is None:
        raise ValueError('Calibration file is missing camera_matrix or distortion_coefficients')
    rvecs = data.get('rotation_vectors')
    tvecs = data.get('translation_vectors')
    return mtx, dist, data, rvecs, tvecs

def is_fisheye(dist):
    flat = np.array(dist).reshape(-1)
    return flat.size == 4

def collect_files(inputs):
    files = []
    for pattern in inputs:
        expanded = glob.glob(pattern, recursive=True) or [pattern]
        for item in expanded:
            if os.path.isdir(item):
                for root, _, names in os.walk(item):
                    for name in names:
                        if Path(name).suffix.lower() in SUPPORTED_EXTS:
                            files.append(os.path.join(root, name))
            else:
                if Path(item).suffix.lower() in SUPPORTED_EXTS:
                    files.append(item)
    return sorted({os.path.abspath(f) for f in files})

def print_calibration_info(calibration_path, mtx, dist, model, data, rvecs, tvecs):
    with np.printoptions(precision=4, suppress=True):
        print(f'Calibration file: {calibration_path}')
        print(f'Model: {model}')
        print(f'Camera matrix:\n{mtx}')
        print(f'Distortion coefficients: {dist.ravel()}')
        reproj = data.get('reprojection_error')
        if reproj is not None:
            print(f'RMS reprojection error: {reproj}')
        if rvecs is not None:
            print(f'Rotation vectors:\n{rvecs}')
        if tvecs is not None:
            print(f'Translation vectors:\n{tvecs}')

def undistort_image(image, mtx, dist, fisheye, alpha=1.0, balance=0.0, crop=False, zoom_factor=1.0):
    h, w = image.shape[:2]
    if fisheye:
        new_mtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, (w, h), np.eye(3), balance=balance)

        # Apply manual zoom correction
        mtx_zoomed = new_mtx.copy()
        mtx_zoomed[0, 0] = new_mtx[0, 0] * zoom_factor # fx
        mtx_zoomed[1, 1] = new_mtx[1, 1] * zoom_factor # fy

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx_zoomed, (w, h), cv2.CV_16SC2)
        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))
    undistorted = cv2.undistort(image, mtx, dist, None, new_mtx)
    if crop and roi is not None:
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y + h_roi, x:x + w_roi]
    return undistorted

def main():
    parser = argparse.ArgumentParser(description='Undistort one or more images using saved calibration data.')
    parser.add_argument('inputs', nargs='*', help='Image files, directories, or glob patterns to undistort.')
    parser.add_argument('-c', '--calibration-file', default=CALIBRATION_FILE, help='Path to calibration pickle file.')
    parser.add_argument('-o', '--output-dir', default='output/undistorted_offline', help='Directory to save undistorted images.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Free scaling parameter for standard model (0..1).')
    parser.add_argument('--balance', type=float, default=0.0, help='Fisheye balance parameter (0..1).')
    parser.add_argument('--crop', action='store_true', help='Crop to valid ROI (standard model only).')
    parser.add_argument('--info', action='store_true', help='Print calibration info and exit without processing images.')
    parser.add_argument('--zoom_factor', type=float, default=1.0, help='Zoom factor for fisheye undistortion (>1.0 zooms in).')
    args = parser.parse_args()

    try:
        mtx, dist, data, rvecs, tvecs = load_calibration(args.calibration_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f'Error: {exc}')
        return

    fisheye = is_fisheye(dist)
    model = 'fisheye' if fisheye else 'pinhole'

    print(f'Using {model} camera model for undistortion. Zoom factor: {args.zoom_factor}')

    if args.info:
        print_calibration_info(args.calibration_file, mtx, dist, model, data, rvecs, tvecs)
        return

    files = collect_files(args.inputs)
    if not files:
        print('No input images found. Supported extensions: ' + ', '.join(sorted(SUPPORTED_EXTS)))
        return

    print(f'Loaded calibration from {args.calibration_file} ({model} model)')
    print(f'Processing {len(files)} image(s)...')

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, path in enumerate(files, 1):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            print(f'[{idx}/{len(files)}] Skipped {path} (could not read)')
            continue
        undistorted = undistort_image(img, mtx, dist, fisheye=fisheye, alpha=args.alpha, balance=args.balance, crop=args.crop, zoom_factor=args.zoom_factor)
        out_name = f"undistorted_{Path(path).name}"
        out_path = os.path.join(args.output_dir, out_name)
        success = cv2.imwrite(out_path, undistorted)
        status = 'Saved' if success else 'Failed to save'
        print(f'[{idx}/{len(files)}] {status} {out_path}')

if __name__ == '__main__':
    main()
