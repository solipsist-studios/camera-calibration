#!/usr/bin/env python3
"""Calibrate one camera from checkerboard footage, from the command line.

    python calibrate_video.py footage.mp4 --camera 0006 --out-dir calib/ [--model fisheye]

Reads every STRIDE-th frame, finds the board, keeps a spread of distinct views, calibrates, and
writes ``Camera_<camera>.pkl`` (the same dictionary ``camera_calibration.py`` writes, plus
``model`` and ``image_size``) and a JSON result with the RMS reprojection error and how many
boards were found. Folders of still images work too: pass the folder instead of a video.

Exit status: 0 on success, 1 when too few boards were found (the JSON says so), 2 on bad input.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DETECT_MAX_SIDE = 1600  # corners are found on a reduced copy, then refined on the full frame


def log(message: str) -> None:
    print(message, flush=True)


def frames(source: Path, stride: int, max_frames: int | None):
    """Yield (index, BGR image) for every `stride`-th frame of a video, or of a folder of images."""
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for i, path in enumerate(files[::stride]):
            if max_frames is not None and i >= max_frames:
                return
            image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is not None:
                yield i * stride, image
        return
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {source}")
    index = emitted = 0
    try:
        while True:
            ok = capture.grab()
            if not ok:
                return
            if index % stride == 0:
                ok, image = capture.retrieve()
                if ok:
                    yield index, image
                    emitted += 1
                    if max_frames is not None and emitted >= max_frames:
                        return
            index += 1
    finally:
        capture.release()


def find_board(gray: np.ndarray, pattern: tuple[int, int]):
    """Corner positions (N, 1, 2) in full-frame pixels, or None."""
    h, w = gray.shape
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(small, pattern, flags)
    if not found:
        found, corners = cv2.findChessboardCornersSB(small, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            return None
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2) / scale
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def spread(candidates: list, keep: int, min_move_px: float) -> list:
    """Drop views that barely moved since the last kept one, then take `keep` evenly spaced ones."""
    distinct = []
    for item in candidates:
        if not distinct or np.linalg.norm(item["corners"] - distinct[-1]["corners"], axis=2).mean() >= min_move_px:
            distinct.append(item)
    if len(distinct) <= keep:
        return distinct
    return [distinct[int(i)] for i in np.linspace(0, len(distinct) - 1, keep).round()]


def calibrate(model: str, objp: np.ndarray, views: list, size: tuple[int, int]):
    """Returns (rms, K, D, rvecs, tvecs, kept views). Fisheye drops views OpenCV calls ill-conditioned."""
    views = list(views)
    while True:
        obj = [objp for _ in views]
        img = [v["corners"] for v in views]
        if model == "fisheye":
            K, D = np.zeros((3, 3)), np.zeros((4, 1))
            flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
            try:
                rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    [o.reshape(1, -1, 3) for o in obj], [i.reshape(1, -1, 2) for i in img], size, K, D, flags=flags)
                return rms, K, D, rvecs, tvecs, views
            except cv2.error as error:
                text = str(error)
                if "input array" in text and "ill-conditioned" in text.lower() and len(views) > 8:
                    bad = int(text.split("input array")[1].split()[0])
                    log(f"dropping view {bad} (ill-conditioned)")
                    views.pop(bad)
                    continue
                raise
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, None, None)
        return rms, K, D, rvecs, tvecs, views


def per_view_errors(model: str, objp, views, K, D, rvecs, tvecs) -> list[float]:
    out = []
    for view, r, t in zip(views, rvecs, tvecs):
        if model == "fisheye":
            projected, _ = cv2.fisheye.projectPoints(objp.reshape(1, -1, 3), r, t, K, D)
        else:
            projected, _ = cv2.projectPoints(objp, r, t, K, D)
        diff = view["corners"].reshape(-1, 2) - projected.reshape(-1, 2)
        out.append(float(np.sqrt((diff ** 2).sum(axis=1).mean())))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="checkerboard video, or a folder of images")
    p.add_argument("--camera", required=True, help="camera id; the output is Camera_<id>.pkl")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", choices=("opencv", "fisheye"), default="fisheye")
    p.add_argument("--board-cols", type=int, default=9, help="inner corners per row")
    p.add_argument("--board-rows", type=int, default=6, help="inner corners per column")
    p.add_argument("--square-cm", type=float, default=4.3)
    p.add_argument("--stride", type=int, default=15, help="read every Nth frame")
    p.add_argument("--max-frames", type=int, default=None, help="stop after reading this many frames")
    p.add_argument("--keep", type=int, default=40, help="calibrate from at most this many distinct views")
    p.add_argument("--min-move-px", type=float, default=15.0, help="a view must move this much (mean corner px) to count as distinct")
    p.add_argument("--min-boards", type=int, default=10, help="fail when fewer distinct views than this were found")
    p.add_argument("--json", type=Path, default=None, help="where to write the result (default OUT_DIR/Camera_<id>.json)")
    args = p.parse_args(argv)
    if args.stride < 1 or args.board_cols < 3 or args.board_rows < 3:
        print("stride must be at least 1 and the board at least 3x3 corners", file=sys.stderr)
        return 2
    if not args.source.exists():
        print(f"{args.source} does not exist", file=sys.stderr)
        return 2

    pattern = (args.board_cols, args.board_rows)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * args.square_cm
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.json or args.out_dir / f"Camera_{args.camera}.json"

    size = None
    read = 0
    found = []
    for index, image in frames(args.source, args.stride, args.max_frames):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != size:
            continue
        read += 1
        corners = find_board(gray, pattern)
        if corners is not None:
            found.append({"frame": index, "corners": corners})
        if read % 10 == 0:
            log(f"read {read} frames, {len(found)} with the board")
    result = {"camera": args.camera, "model": args.model.upper() if args.model == "fisheye" else "OPENCV",
              "model_name": "OPENCV_FISHEYE" if args.model == "fisheye" else "OPENCV", "image_size": list(size) if size else None,
              "frames_read": read, "boards_found": len(found), "boards_used": 0, "stride": args.stride,
              "board": {"cols": pattern[0], "rows": pattern[1], "square_cm": args.square_cm}, "ok": False}
    result["model"] = result.pop("model_name")
    views = spread(found, args.keep, args.min_move_px)
    result["distinct_views"] = len(views)
    if len(views) < args.min_boards:
        result["error"] = (f"only {len(views)} distinct views of the board were found in {read} frames "
                           f"({len(found)} with the board); need at least {args.min_boards}")
        result_path.write_text(json.dumps(result, indent=2))
        log(result["error"])
        return 1

    log(f"calibrating {args.camera} from {len(views)} views ({args.model})")
    rejected = 0
    for _ in range(4):
        rms, K, D, rvecs, tvecs, kept = calibrate(args.model, objp, views, size)
        errors = per_view_errors(args.model, objp, kept, K, D, rvecs, tvecs)
        # A blurred or mis-detected board shows as a view far worse than the rest; drop it and solve again.
        limit = max(3.0 * float(np.median(errors)), 1.5)
        good = [v for v, e in zip(kept, errors) if e <= limit]
        if len(good) == len(kept) or len(good) < max(args.min_boards, 8):
            break
        rejected += len(kept) - len(good)
        log(f"dropping {len(kept) - len(good)} view(s) with error above {limit:.2f} px")
        views = good
    data = {"model": result["model"], "camera_matrix": K, "distortion_coefficients": D,
            "rotation_vectors": tuple(rvecs), "translation_vectors": tuple(tvecs),
            "reprojection_error": float(rms), "image_size": size}
    target = args.out_dir / f"Camera_{args.camera}.pkl"
    with open(target.with_suffix(".tmp"), "wb") as f:
        pickle.dump(data, f)
    os.replace(target.with_suffix(".tmp"), target)
    result.update(ok=True, boards_used=len(kept), views_rejected=rejected, reprojection_error=float(rms), per_view_error=errors,
                  camera_matrix=K.tolist(), distortion_coefficients=D.ravel().tolist(), file=str(target))
    result_path.write_text(json.dumps(result, indent=2))
    log(f"RMS reprojection error {rms:.4f} px from {len(kept)} views -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
