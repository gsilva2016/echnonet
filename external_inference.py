#!/usr/bin/env python3
"""Run EchoNet pretrained inference on external video files (e.g., .mp4).

This script does not require FileList.csv or VolumeTracings.csv.
It loads videos from a folder, runs:
1) EF prediction (R2Plus1D) on all clips, and
2) frame-wise LV segmentation (DeepLabV3),
then writes outputs to an output directory.
"""

import argparse
import os
import time
import torch
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import scipy.signal
import echonet
import openvino as ov


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".m4v", ".mpg", ".mpeg", ".wmv", ".mkv"}
core = ov.Core()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EchoNet external video inference")
    parser.add_argument("--videos_dir", type=str, required=True, help="Directory containing external videos")
    parser.add_argument("--output_dir", type=str, default="output/external_inference", help="Output directory")
    parser.add_argument("--ef_model", type=str, required=False, help="Path to EF OpenVINO model (.xml)")
    parser.add_argument("--seg_model", type=str, required=True, help="Path to segmentation OpenVINO model (.xml)")
    parser.add_argument("--frames", type=int, default=32, help="Number of frames per EF clip")
    parser.add_argument("--period", type=int, default=2, help="Frame sampling period for EF clips")
    parser.add_argument("--max_length", type=int, default=250, help="Max frames for full-video segmentation")
    parser.add_argument("--clip_batch_size", type=int, default=32, help="EF clip batch size per forward pass")
    parser.add_argument("--seg_batch_size", type=int, default=64, help="Segmentation frame batch size")
    parser.add_argument("--fps", type=float, default=50.0, help="FPS for saved segmentation overlays")
    parser.add_argument("--device", type=str, default="GPU", help="OpenVINO device, e.g. GPU, NPU, or CPU")
    parser.add_argument("--max_videos", type=int, default=None, help="Optional cap on number of videos to process")
    parser.add_argument("--display", action="store_true", help="Show live split-screen preview during segmentation")
    parser.add_argument("--display_scale", type=float, default=4.0, help="Scale factor for preview window size")
    parser.add_argument("--display_hold_ms", type=int, default=5000, help="How long to keep preview open after playback (0 = wait for key)")
    parser.add_argument("--display_loops", type=int, default=10000, help="How many times to replay each preview clip")
    return parser.parse_args()


def list_videos(videos_dir: str) -> List[Path]:
    root = Path(videos_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory not found: {videos_dir}")
    videos = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        raise FileNotFoundError(f"No supported video files found in: {videos_dir}")
    return videos

def compute_mean_std(video_paths: Sequence[Path], sample_count: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    if len(video_paths) > sample_count:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(video_paths), size=sample_count, replace=False)
        sample_paths = [video_paths[i] for i in indices]
    else:
        sample_paths = list(video_paths)

    s1 = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)
    n = 0

    for path in sample_paths: # tqdm.tqdm(sample_paths, desc="Computing mean/std"):
        video = echonet.utils.loadvideo(str(path)).astype(np.float32)  # (3, f, h, w)
        x = video.reshape(3, -1)
        s1 += x.sum(axis=1)
        s2 += (x ** 2).sum(axis=1)
        n += x.shape[1]

    mean = (s1 / n).astype(np.float32)
    std = np.sqrt(s2 / n - mean ** 2).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def build_ef_model(model_path: str, device: str):
    model = core.read_model(model_path)

    model = core.compile_model(model, device)
    return model

def build_seg_model(model_path: str, device: str):
    model = core.read_model(model_path)
    #model.reshape("56..64, 3, 112, 112")
    model = core.compile_model(model, device)
    return model


def resize_video(video: np.ndarray, size: int = 112) -> np.ndarray:
    # video: (3, f, h, w) -> resized to (3, f, size, size)
    frames = video.transpose(1, 2, 3, 0)
    resized = np.stack([
        cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
        for frame in frames
    ], axis=0)
    return resized.transpose(3, 0, 1, 2).astype(np.float32)


def run_ef_inference(
    model: torch.nn.Module,
    video_paths: Sequence[Path],
    mean: np.ndarray,
    std: np.ndarray,
    frames: int,
    period: int,
    clip_batch_size: int,
    device: str,
    output_dir: Path,
) -> None:
    output_file = output_dir / "ef_predictions.csv"
    with output_file.open("w", encoding="utf-8") as f:
        f.write("Filename,ClipIndex,PredictedEF,MeanPredictedEF\n")

        for path in video_paths: #tqdm.tqdm(video_paths, desc="EF inference"):
            video = echonet.utils.loadvideo(str(path)).astype(np.float32)
            video = resize_video(video, size=112)
            video -= mean.reshape(3, 1, 1, 1)
            video /= std.reshape(3, 1, 1, 1)

            c, f_count, h, w = video.shape
            required = frames * period
            if f_count < required:
                pad = np.zeros((c, required - f_count, h, w), dtype=video.dtype)
                video = np.concatenate((video, pad), axis=1)
                f_count = video.shape[1]

            starts = np.arange(f_count - (frames - 1) * period)
            clip_preds = []

            with torch.no_grad():
                start_t = time.time()
                for j in range(0, len(starts), clip_batch_size):
                    s_batch = starts[j:(j + clip_batch_size)]
                    batch_np = np.stack([
                        video[:, s + period * np.arange(frames), :, :]
                        for s in s_batch
                    ], axis=0)
                    batch = torch.from_numpy(batch_np)
                    pred = model(batch)["x.33"]
                    clip_preds.extend(pred.tolist())
            print("EF Inference Time Taken: ", time.time() - start_t)

            clip_preds = np.asarray(clip_preds, dtype=np.float32)
            mean_pred = float(np.mean(clip_preds))

            for i, pred in enumerate(clip_preds):
                #f.write(f"{path.name},{i},{str(pred):.4f},{mean_pred:.4f}\n")
                f.write(f"{path.name},{i},{str(pred)},{mean_pred:.4f}\n")


def safe_peaks(size: np.ndarray) -> set:
    if size.size < 3:
        return set()
    trim_min = sorted(size)[round(len(size) ** 0.05)]
    trim_max = sorted(size)[round(len(size) ** 0.95)]
    trim_range = trim_max - trim_min
    prominence = max(1.0, 0.50 * trim_range)
    peaks = scipy.signal.find_peaks(-size, distance=20, prominence=prominence)[0]
    return set(int(p) for p in peaks)


def run_segmentation_inference(
    model: torch.nn.Module,
    video_paths: Sequence[Path],
    mean: np.ndarray,
    std: np.ndarray,
    seg_batch_size: int,
    device: str,
    output_dir: Path,
    fps: float,
    max_length: int,
    display: bool,
    display_scale: float,
    display_hold_ms: int,
    display_loops: int,
) -> None:
    videos_out = output_dir / "segmentation_videos"
    size_out = output_dir / "segmentation_size"
    videos_out.mkdir(parents=True, exist_ok=True)
    size_out.mkdir(parents=True, exist_ok=True)

    size_csv = output_dir / "size.csv"
    with size_csv.open("w", encoding="utf-8") as g:
        g.write("Filename,Frame,Size,ComputerSmall\n")

        window_name = "EchoNet Preview | Left: Original | Right: Segmentation"
        should_display = display
        if should_display:
            print("Opening preview window. Press 'q' to close preview playback.")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        for path in video_paths: #tqdm.tqdm(video_paths, desc="Segmentation inference"):
            raw_full = echonet.utils.loadvideo(str(path)).astype(np.float32)  # (3, f, h, w)
            if max_length is not None and raw_full.shape[1] > max_length:
                raw_full = raw_full[:, :max_length, :, :]

            # Run segmentation at model resolution, then map masks back to full resolution.
            raw_small = resize_video(raw_full, size=112)

            video = raw_small.copy()
            video -= mean.reshape(3, 1, 1, 1)
            video /= std.reshape(3, 1, 1, 1)

            frames_first = torch.from_numpy(video.transpose(1, 0, 2, 3))  # (f, 3, h, w)
            logits = []
            infer_start = time.perf_counter()
            with torch.no_grad():
                start_t = time.time()
                for i in range(0, frames_first.shape[0], seg_batch_size):
                    batch = frames_first[i:(i + seg_batch_size)]
                    y = model(batch)["output"][:, 0, :, :]
                    logits.append(y)
            print("Seg inference time taken: ", time.time() - start_t)
            infer_end = time.perf_counter()
            logit = np.concatenate(logits, axis=0)

            mask_small = (logit > 0).astype(np.uint8)
            f_count = mask_small.shape[0]
            full_h, full_w = raw_full.shape[2], raw_full.shape[3]
            mask = np.stack([
                cv2.resize(mask_small[i], (full_w, full_h), interpolation=cv2.INTER_NEAREST)
                for i in range(f_count)
            ], axis=0).astype(np.uint8)
            size = mask.sum(axis=(1, 2)).astype(np.int64)
            systole = safe_peaks(size)

            for frame, s in enumerate(size):
                g.write(f"{path.name},{frame},{int(s)},{1 if frame in systole else 0}\n")

            # Create side-by-side overlay video in RGB space at full resolution.
            disp = np.clip(raw_full, 0, 255).astype(np.uint8)
            _, h, w = mask.shape
            side = np.concatenate((disp, disp), axis=3)  # (3, f, h, 2w)
            # Saturate blue channel on right half for segmentation.
            side[0, :, :, w:] = np.maximum(side[0, :, :, w:], (255 * mask).astype(np.uint8))

            if should_display:
                delay_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))
                infer_seconds = max(1e-9, infer_end - infer_start)
                infer_fps = f_count / infer_seconds
                infer_latency_ms = (infer_seconds / max(1, f_count)) * 1000.0
                playback_fps = max(1e-9, fps)
                playback_latency_ms = 1000.0 / playback_fps
                prev_tick = None
                ema_display_fps = None

                loops = max(1, display_loops)
                for _ in range(loops):
                    for frame_idx in range(f_count):
                        frame_rgb = side[:, frame_idx, :, :].transpose(1, 2, 0)
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                        now = time.perf_counter()
                        if prev_tick is not None:
                            dt = max(1e-9, now - prev_tick)
                            instant_fps = 1.0 / dt
                            if ema_display_fps is None:
                                ema_display_fps = instant_fps
                            else:
                                ema_display_fps = 0.9 * ema_display_fps + 0.1 * instant_fps
                        prev_tick = now

                        display_fps_text = ema_display_fps if ema_display_fps is not None else playback_fps
                        display_latency_ms = 1000.0 / max(1e-9, display_fps_text)

                        cv2.putText(
                            frame_bgr,
                            f"Infer throughput: {infer_fps:.2f} fps",
                            (20, 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame_bgr,
                            f"Infer latency: {infer_latency_ms:.2f} ms/frame",
                            (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame_bgr,
                            f"Display throughput: {display_fps_text:.2f} fps",
                            (20, 88),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame_bgr,
                            f"Display latency: {display_latency_ms:.2f} ms/frame (target {playback_latency_ms:.2f})",
                            (20, 116),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        if display_scale and display_scale > 0:
                            h2, w2 = frame_bgr.shape[:2]
                            frame_bgr = cv2.resize(
                                frame_bgr,
                                (int(w2 * display_scale), int(h2 * display_scale)),
                                interpolation=cv2.INTER_NEAREST,
                            )

                        cv2.imshow(window_name, frame_bgr)
                        # Press q to close live preview while inference continues.
                        key = cv2.waitKey(delay_ms) & 0xFF
                        if key == ord("q"):
                            should_display = False
                            cv2.destroyWindow(window_name)
                            break
                    if not should_display:
                        break

                if should_display:
                    final = cv2.cvtColor(side[:, -1, :, :].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    if display_scale and display_scale > 0:
                        h2, w2 = final.shape[:2]
                        final = cv2.resize(
                            final,
                            (int(w2 * display_scale), int(h2 * display_scale)),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    cv2.putText(
                        final,
                        "Preview complete - press any key to continue",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, final)
                    if display_hold_ms == 0:
                        cv2.waitKey(0)
                    else:
                        cv2.waitKey(display_hold_ms)

            out_name = f"{path.stem}.avi"
            echonet.utils.savevideo(str(videos_out / out_name), side, fps=fps)

        if display and should_display:
            cv2.destroyWindow(window_name)


def main() -> None:
    args = parse_args()

    videos = list_videos(args.videos_dir)
    if args.max_videos is not None:
        videos = videos[:max(0, args.max_videos)]
    if not videos:
        raise ValueError("No videos selected after applying --max_videos")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")
    print(f"Found {len(videos)} videos in {args.videos_dir}")

    mean, std = compute_mean_std(videos)
    np.save(output_dir / "mean.npy", mean)
    np.save(output_dir / "std.npy", std)
    print(f"Computed mean: {mean}")
    print(f"Computed std:  {std}")

    if args.ef_model:
        ef_model = build_ef_model(args.ef_model, device)    
        start_ef_t = time.time()
        run_ef_inference(
            ef_model,
            videos,
            mean,
            std,
            frames=args.frames,
            period=args.period,
            clip_batch_size=args.clip_batch_size,
            device=device,
            output_dir=output_dir,
        )
        print("Total EF Inference time: ", time.time() - start_ef_t)
        print(f"Wrote EF predictions to {output_dir / 'ef_predictions.csv'}")
    
    seg_model = build_seg_model(args.seg_model, device)
    start_seg_t = time.time()
    run_segmentation_inference(
        seg_model,
        videos,
        mean,
        std,
        seg_batch_size=args.seg_batch_size,
        device=device,
        output_dir=output_dir,
        fps=args.fps,
        max_length=args.max_length,
        display=args.display,
        display_scale=args.display_scale,
        display_hold_ms=args.display_hold_ms,
        display_loops=args.display_loops,
    )
    print("Total Segmentation Inference time: ", time.time() - start_seg_t)
    print(f"Wrote segmentation outputs under {output_dir}")


if __name__ == "__main__":
    main()
