import argparse
import glob
import os
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

CALIBRATION_PATH = 'output/calibration_data.pkl'
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
SUPPORTED_MODELS = {'OPENCV', 'OPENCV_FISHEYE'}


def load_matrix_from_text(path):
    matrix = np.loadtxt(path)
    matrix = np.array(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f'camera_matrix must be 3x3, got shape {matrix.shape}')
    return matrix


def load_distortion_from_text(path):
    dist = np.loadtxt(path)
    dist = np.array(dist, dtype=np.float64).reshape(-1)
    if dist.size not in {4, 5, 8, 12, 14}:
        raise ValueError(
            'distortion_coefficients must contain 4, 5, 8, 12, or 14 values '
            f'(got {dist.size})'
        )
    return dist.reshape(1, -1)


def ensure_directory(path):
    if path:

        os.makedirs(path, exist_ok=True)



def resolve_output_target(output_path, is_single_input):
    if is_single_input and output_path and Path(output_path).suffix:
        return output_path, os.path.dirname(output_path)
    return output_path, output_path


def save_calibration_from_text(camera_matrix_path, distortion_coefficients_path, output_path, model=None):
    if not os.path.exists(camera_matrix_path):
        raise FileNotFoundError(f'Camera matrix file not found at {camera_matrix_path}')
    if not os.path.exists(distortion_coefficients_path):
        raise FileNotFoundError(f'Distortion coefficients file not found at {distortion_coefficients_path}')

    mtx = load_matrix_from_text(camera_matrix_path)
    dist = load_distortion_from_text(distortion_coefficients_path)

    inferred_model = infer_model(model, dist)
    calibration_data = {
        'camera_matrix': mtx,
        'distortion_coefficients': dist,
        'rotation_vectors': None,
        'translation_vectors': None,
        'reprojection_error': None,
        'image_size': None,
        'model': inferred_model,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        ensure_directory(output_dir)
    with open(output_path, 'wb') as f:
        pickle.dump(calibration_data, f)

    return inferred_model, mtx, dist

def infer_model(model, dist):
    if model in SUPPORTED_MODELS:
        return model

    # Backward compatibility for older calibration files that do not include model.
    flat = np.array(dist).reshape(-1)
    return 'OPENCV_FISHEYE' if flat.size == 4 else 'OPENCV'

def load_calibration(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Calibration file not found at {path}')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    mtx = data.get('camera_matrix')
    dist = data.get('distortion_coefficients')
    if mtx is None or dist is None:
        raise ValueError('Calibration file is missing camera_matrix or distortion_coefficients')
    model_from_file = data.get('model')
    model = infer_model(model_from_file, dist)
    rvecs = data.get('rotation_vectors')
    tvecs = data.get('translation_vectors')
    image_size = data.get('image_size')
    if image_size is not None:
        width, height = image_size
    else:
        width = height = None
    return width, height, mtx, dist, model, data, rvecs, tvecs


def resolve_calibration_for_image(calibration_path, rel_path):
    """Resolve calibration file for an image.

    If calibration_path is a file, that file is used for all images.
    If calibration_path is a directory, find a .pkl file that matches image naming.
    """
    if os.path.isfile(calibration_path):
        return calibration_path

    if not os.path.isdir(calibration_path):
        raise FileNotFoundError(f'Calibration path not found: {calibration_path}')

    rel_stem = str(Path(rel_path).with_suffix('.pkl'))
    basename_stem = f'{Path(rel_path).stem}.pkl'

    candidates = [
        os.path.join(calibration_path, rel_stem),
        os.path.join(calibration_path, basename_stem),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        'No matching calibration .pkl found for image '
        f'"{rel_path}" in directory "{calibration_path}". '
        f'Tried: {", ".join(candidates)}'
    )

def print_calibration_info(calibration_path, width, height, mtx, dist, model, data, rvecs, tvecs):
    with np.printoptions(precision=16, suppress=True):
        print(f'Calibration file: {calibration_path}')
        print(f'Model: {model}')
        print(f'Image size: {width}x{height}')
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
        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))
    undistorted = cv2.undistort(image, mtx, dist, None, new_mtx)
    if crop and roi is not None:
        x, y, w_roi, h_roi = roi
        undistorted = undistorted[y:y + h_roi, x:x + w_roi]
    return undistorted


def build_undistorted_calibration_data(mtx, dist, image_size, fisheye, alpha=1.0, balance=0.0, zoom_factor=1.0):
    w, h = image_size
    if fisheye:
        new_mtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            mtx, dist, (w, h), np.eye(3), balance=balance
        )
        new_mtx[0, 0] = new_mtx[0, 0] * zoom_factor  # fx
        new_mtx[1, 1] = new_mtx[1, 1] * zoom_factor  # fy
    else:
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))

    return {
        'camera_matrix': new_mtx,
        'distortion_coefficients': np.zeros_like(dist),
        'rotation_vectors': None,
        'translation_vectors': None,
        'reprojection_error': None,
        'image_size': (w, h),
        'model': 'OPENCV',
        'undistortion_parameters': {
            'alpha': alpha if not fisheye else None,
            'balance': balance if fisheye else None,
            'zoom_factor': zoom_factor if fisheye else None,
        },
    }

def collect_files_with_structure(inputs):
    """Collect files while preserving their relative paths from source directories."""
    files = []  # List of (absolute_path, relative_path, source_base_dir)
    for pattern in inputs:
        expanded = glob.glob(pattern, recursive=True) or [pattern]
        for item in expanded:
            if os.path.isdir(item):
                base_dir = os.path.abspath(item)
                for root, _, names in os.walk(item):
                    for name in names:
                        if Path(name).suffix.lower() in SUPPORTED_EXTS:
                            full_path = os.path.join(root, name)
                            rel_path = os.path.relpath(full_path, base_dir)
                            files.append((full_path, rel_path, base_dir))
            else:
                if Path(item).suffix.lower() in SUPPORTED_EXTS:
                    files.append((os.path.abspath(item), Path(item).name, None))
    return sorted(files, key=lambda x: x[0])

def process_image_worker(args):
    """Process a single image (worker function for parallel processing)."""
    (
        abs_path,
        rel_path,
        source_base,
        calib_path,
        output_dir,
        fisheye,
        alpha,
        balance,
        crop,
        zoom_factor,
        undistorted_calibration_out,
        output_target,
    ) = args
    
    try:
        # Load calibration data
        _, _, mtx, dist, _, _, _, _ = load_calibration(calib_path)
        
        # Read image
        img = cv2.imread(abs_path, flags=cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            return (abs_path, False, 'Could not read')
        h, w = img.shape[:2]
        
        # Undistort image
        undistorted = undistort_image(img, mtx, dist, fisheye=fisheye, alpha=alpha, 
                                     balance=balance, crop=crop, zoom_factor=zoom_factor)
        if output_target and Path(output_target).suffix:
        # Build output path preserving directory structure
        if output_target and Path(output_target).suffix and len(rel_path) > 0 and source_base is None:
            out_dir = os.path.dirname(output_target)
            out_name = os.path.basename(output_target)
        elif source_base is not None:
            dir_part = os.path.dirname(rel_path)
            file_part = Path(rel_path).name
            out_dir = os.path.join(output_dir, dir_part) if dir_part else output_dir
            out_name = f"undistorted_{file_part}"
        else:
            out_dir = output_dir
            out_name = f"undistorted_{rel_path}"
        
        ensure_directory(out_dir)
        out_path = os.path.join(out_dir, out_name)
        success = cv2.imwrite(out_path, undistorted)

        if success and undistorted_calibration_out is not None:
            undistorted_data = build_undistorted_calibration_data(
                mtx,
                dist,
                (w, h),
                fisheye,
                alpha=alpha,
                balance=balance,
                zoom_factor=zoom_factor,
            )
            calib_dir = os.path.dirname(undistorted_calibration_out)
            if calib_dir:
                ensure_directory(calib_dir)
            with open(undistorted_calibration_out, 'wb') as f:
                pickle.dump(undistorted_data, f)
        
        return (out_path, success, None)
    except Exception as e:
        return (abs_path, False, str(e))

def main():
    parser = argparse.ArgumentParser(description='Undistort one or more images using saved calibration data.')
    parser.add_argument('inputs', nargs='*', help='Image files, directories, or glob patterns to undistort.')
    parser.add_argument(
        '-c',
        '--calibration-path',
        default=CALIBRATION_PATH,
        help='Path to a calibration .pkl file, or a directory of per-image .pkl files named like the input images.',
    )
    parser.add_argument('-o', '--output-dir', default='output/undistorted_offline', help='Directory to save undistorted images.')
    parser.add_argument(
        '-u',
        '--undistorted-calibration',
        type=str,
        help=(
            'Path for undistorted calibration output. In single-calibration mode, this is a file path. '
            'In batch mode, this is treated as a directory for per-image calibration files.'
        ),
    )
    parser.add_argument('--alpha', type=float, default=1.0, help='Free scaling parameter for standard model (0..1).')
    parser.add_argument('--balance', type=float, default=0.0, help='Fisheye balance parameter (0..1).')
    parser.add_argument('--crop', action='store_true', help='Crop to valid ROI (standard model only).')
    parser.add_argument('--info', action='store_true', help='Print calibration info and exit without processing images.')
    parser.add_argument('--zoom_factor', type=float, default=1.0, help='Zoom factor for fisheye undistortion (>1.0 zooms in).')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: number of CPU cores).')
    parser.add_argument('--generate-calibration-pkl', type=str, help='Generate a calibration .pkl file from text intrinsics and exit.')
    parser.add_argument('--camera-matrix-file', type=str, help='Path to camera_matrix text file (for --generate-calibration-pkl mode).')
    parser.add_argument('--distortion-coefficients-file', type=str, help='Path to distortion_coefficients text file (for --generate-calibration-pkl mode).')
    parser.add_argument('--model', choices=sorted(SUPPORTED_MODELS), help='Optional camera model to store in generated calibration file.')
    args = parser.parse_args()

    if args.generate_calibration_pkl:
        if not args.camera_matrix_file or not args.distortion_coefficients_file:
            print('Error: --camera-matrix-file and --distortion-coefficients-file are required with --generate-calibration-pkl')
            return
        try:
            model, mtx, dist = save_calibration_from_text(
                args.camera_matrix_file,
                args.distortion_coefficients_file,
                args.generate_calibration_pkl,
                args.model,
            )
            print(f'Saved calibration pkl to {args.generate_calibration_pkl}')
            print(f'Model: {model}')
            print(f'Camera matrix shape: {mtx.shape}, distortion shape: {dist.shape}')
        except (FileNotFoundError, ValueError) as exc:
            print(f'Error: {exc}')
        return

    files = collect_files_with_structure(args.inputs)

    is_calibration_dir = os.path.isdir(args.calibration_path)

    if is_calibration_dir and args.info:
        print('Error: --info is only supported when --calibration-path is a single .pkl file.')
        return

    if is_calibration_dir and args.undistorted_calibration and os.path.isfile(args.undistorted_calibration):
        print('Error: in batch mode, --undistorted-calibration must be a directory path, not a file path.')
        return

    if is_calibration_dir:
        model = 'MIXED'
        data = {}
        rvecs = None
        tvecs = None
        fisheye = False
        print('Using per-image calibration files from directory: ' + args.calibration_path)
    else:
        try:
            w, h, mtx, dist, model, data, rvecs, tvecs = load_calibration(args.calibration_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f'Error: {exc}')
            return
        fisheye = model == 'OPENCV_FISHEYE'
        print(f'Using {model} camera model for undistortion. Zoom factor: {args.zoom_factor}')

    if args.undistorted_calibration and not is_calibration_dir:
        # Prefer image size from calibration metadata; fall back to first input image if absent
        calib_size = data.get('image_size')
        w = h = None
        if calib_size is not None and len(calib_size) == 2:
            w, h = calib_size
        if (w is None or h is None) and files:
            first_img = cv2.imread(files[0][0], flags=cv2.IMREAD_COLOR + cv2.IMREAD_IGNORE_ORIENTATION)
            if first_img is not None:
                h, w = first_img.shape[:2]
            else:
                print('Could not determine image size from first image; skipping undistorted calibration save.')

        if w is not None and h is not None:
            undistorted_data = build_undistorted_calibration_data(
                mtx,
                dist,
                (w, h),
                fisheye,
                alpha=args.alpha,
                balance=args.balance,
                zoom_factor=args.zoom_factor,
            )

            calib_dir = os.path.dirname(args.undistorted_calibration)
            if calib_dir:
                ensure_directory(calib_dir)

            with open(args.undistorted_calibration, 'wb') as f:
                pickle.dump(undistorted_data, f)
            print(f'Saved undistorted calibration to {args.undistorted_calibration}')
        else:
            print('Could not determine image size; skipping undistorted calibration save.')

    if args.info:
        print_calibration_info(args.calibration_path, w, h, mtx, dist, model, data, rvecs, tvecs)
        return

    if not files:
        print('No input images found. Supported extensions: ' + ', '.join(sorted(SUPPORTED_EXTS)))
        return

    if is_calibration_dir:
        print(f'Loaded per-image calibration directory: {args.calibration_path}')
    else:
        print(f'Loaded calibration from {args.calibration_path} ({model} model)')
    print(f'Processing {len(files)} image(s) with {args.workers or "auto"} worker(s)...')

    is_single_input = len(files) == 1
    output_target, output_base_dir = resolve_output_target(args.output_dir, is_single_input)
    ensure_directory(output_base_dir)

    # Prepare work items for parallel processing
    work_items = []
    for abs_path, rel_path, source_base in files:
        try:
            calibration_path = resolve_calibration_for_image(args.calibration_path, rel_path)
        except FileNotFoundError as exc:
            print(f'Error: {exc}')
            failed = len(files)
            print(f'\nCompleted: 0, Failed: {failed}')
            return

        image_fisheye = fisheye
        undistorted_calibration_out = None
        if is_calibration_dir:
            try:
                _, _, _, _, image_model, _, _, _ = load_calibration(calibration_path)
                image_fisheye = image_model == 'OPENCV_FISHEYE'
            except (FileNotFoundError, ValueError) as exc:
                print(f'Error: failed to load calibration "{calibration_path}": {exc}')
                failed = len(files)
                print(f'\nCompleted: 0, Failed: {failed}')
                return

            calib_name = f'undistorted_{Path(rel_path).stem}.pkl'
            if args.undistorted_calibration:
                calib_base_dir = args.undistorted_calibration
                rel_dir = os.path.dirname(rel_path)
                calib_out_dir = os.path.join(calib_base_dir, rel_dir) if rel_dir else calib_base_dir
                undistorted_calibration_out = os.path.join(calib_out_dir, calib_name)
            else:
                rel_dir = os.path.dirname(rel_path) if source_base is not None else ''
                calib_out_dir = os.path.join(args.output_dir, rel_dir) if rel_dir else args.output_dir
                undistorted_calibration_out = os.path.join(calib_out_dir, calib_name)
        elif args.undistorted_calibration:
            undistorted_calibration_out = args.undistorted_calibration

        work_items.append(
            (
                abs_path,
                rel_path,
                source_base,
                calibration_path,
                args.output_dir,
                image_fisheye,
                args.alpha,
                args.balance,
                args.crop,
                args.zoom_factor,
                undistorted_calibration_out,
                output_target,
            )
        )

    # Process images in parallel
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_image_worker, item): i + 1 for i, item in enumerate(work_items)}
        
        with tqdm(total=len(files), desc='Processing', unit='image') as pbar:
            for future in as_completed(futures):
                try:
                    out_path, success, error = future.result()
                    if success:
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                pbar.update(1)
    
    print(f'\nCompleted: {completed}, Failed: {failed}')

if __name__ == '__main__':
    main()
