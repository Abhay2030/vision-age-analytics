import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WEIGHTS = os.path.join(_SCRIPT_DIR, "weights")


@dataclass
class Config:
    face_proto: str
    face_model: str
    age_proto: str
    age_model: str
    confidence_threshold: float = 0.5
    age_buckets: Tuple[str, ...] = (
        "(0-2)",
        "(4-6)",
        "(8-12)",
        "(15-20)",
        "(25-32)",
        "(38-43)",
        "(48-53)",
        "(60-100)",
    )
    model_mean_values: Tuple[float, float, float] = (
        78.4263377603,
        87.7689143744,
        114.895847746,
    )
    age_bucket_centers: Tuple[float, ...] = (1, 5, 10, 18, 28.5, 40.5, 50.5, 75)
    min_face_size_px: int = 60
    track_match_distance_px: float = 90.0
    track_ttl_frames: int = 20
    smooth_alpha: float = 0.25

    @classmethod
    def from_weights_dir(cls, weights_dir: str) -> "Config":
        return cls(
            face_proto=os.path.join(weights_dir, "deploy.prototxt"),
            face_model=os.path.join(
                weights_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel"
            ),
            age_proto=os.path.join(weights_dir, "age_deploy.prototxt"),
            age_model=os.path.join(weights_dir, "age_net.caffemodel"),
        )


@dataclass
class SessionAnalytics:
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    frame_count: int = 0
    total_face_observations: int = 0
    max_faces_in_frame: int = 0
    age_sum: float = 0.0
    age_count: int = 0
    bucket_counter: Counter = field(default_factory=Counter)
    unique_tracks: set = field(default_factory=set)

    def update(self, detections: List[Dict[str, object]]) -> None:
        self.frame_count += 1
        n_faces = len(detections)
        self.total_face_observations += n_faces
        self.max_faces_in_frame = max(self.max_faces_in_frame, n_faces)
        for det in detections:
            self.bucket_counter[det["age_bucket"]] += 1
            self.age_sum += float(det["estimated_age"])
            self.age_count += 1
            self.unique_tracks.add(int(det["track_id"]))

    def avg_age(self) -> float:
        return self.age_sum / self.age_count if self.age_count else 0.0

    def avg_faces_per_frame(self) -> float:
        return self.total_face_observations / self.frame_count if self.frame_count else 0.0

    def summary_dict(self) -> Dict[str, object]:
        return {
            "started_at": self.started_at,
            "frames_processed": self.frame_count,
            "total_face_observations": self.total_face_observations,
            "max_faces_in_frame": self.max_faces_in_frame,
            "avg_faces_per_frame": round(self.avg_faces_per_frame(), 3),
            "estimated_avg_age": round(self.avg_age(), 2),
            "unique_people_tracks": len(self.unique_tracks),
            "bucket_distribution": dict(self.bucket_counter),
        }


class AgeDetectionSystem:
    """Face detection + age estimation with smoothing, tracking and analytics."""

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        use_gpu: bool = False,
        confidence_threshold: Optional[float] = None,
        min_face_size: Optional[int] = None,
        smooth_alpha: Optional[float] = None,
        track_distance: Optional[float] = None,
    ):
        self.cfg = Config.from_weights_dir(weights_dir or _DEFAULT_WEIGHTS)
        if confidence_threshold is not None:
            self.cfg.confidence_threshold = confidence_threshold
        if min_face_size is not None:
            self.cfg.min_face_size_px = min_face_size
        if smooth_alpha is not None:
            self.cfg.smooth_alpha = max(0.01, min(0.99, smooth_alpha))
        if track_distance is not None:
            self.cfg.track_match_distance_px = max(20.0, track_distance)

        self.tracks: Dict[int, Dict[str, object]] = {}
        self.next_track_id = 1
        self.analytics = SessionAnalytics()
        self._check_paths()
        print("[INFO] Loading models...")
        try:
            self.face_net = cv2.dnn.readNet(self.cfg.face_model, self.cfg.face_proto)
            self.age_net = cv2.dnn.readNet(self.cfg.age_model, self.cfg.age_proto)
            self._configure_backend(use_gpu=use_gpu)
        except cv2.error as e:
            raise RuntimeError(
                "Failed to load models. Run `python download_weights.py` or place "
                f"files in {os.path.dirname(self.cfg.face_proto)!r}.\nDetails: {e}"
            ) from e

    def _configure_backend(self, use_gpu: bool) -> None:
        if use_gpu:
            try:
                self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("[INFO] Using CUDA backend.")
                return
            except cv2.error:
                print("[WARN] CUDA unavailable. Falling back to CPU.")
        self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _check_paths(self) -> None:
        required = [
            self.cfg.face_proto,
            self.cfg.face_model,
            self.cfg.age_proto,
            self.cfg.age_model,
        ]
        missing = [p for p in required if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "Missing model files:\n  "
                + "\n  ".join(missing)
                + "\nRun: python download_weights.py"
            )

    def detect_faces(self, frame: np.ndarray) -> List[List[int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        bboxes: List[List[int]] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > self.cfg.confidence_threshold:
                x1 = max(0, int(detections[0, 0, i, 3] * w))
                y1 = max(0, int(detections[0, 0, i, 4] * h))
                x2 = min(w, int(detections[0, 0, i, 5] * w))
                y2 = min(h, int(detections[0, 0, i, 6] * h))
                if x2 > x1 and y2 > y1:
                    bboxes.append([x1, y1, x2, y2])
        return bboxes

    def _enhance_face(self, face_img: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y)
        return cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)

    def _face_quality_score(self, face_img: np.ndarray) -> float:
        """Heuristic quality score in [0,1] based on sharpness and brightness."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        sharp_norm = min(sharpness / 180.0, 1.0)
        bright_norm = 1.0 - min(abs(brightness - 120.0) / 120.0, 1.0)
        return max(0.0, min(1.0, 0.7 * sharp_norm + 0.3 * bright_norm))

    def predict_age_probs(self, face_img: np.ndarray) -> np.ndarray:
        enhanced = self._enhance_face(face_img)
        blob = cv2.dnn.blobFromImage(
            enhanced, 1.0, (227, 227), self.cfg.model_mean_values, swapRB=False
        )
        self.age_net.setInput(blob)
        preds_a = self.age_net.forward()[0].astype(np.float32)

        # Test-time augmentation: horizontal flip often stabilizes age features.
        flipped = cv2.flip(enhanced, 1)
        blob_flip = cv2.dnn.blobFromImage(
            flipped, 1.0, (227, 227), self.cfg.model_mean_values, swapRB=False
        )
        self.age_net.setInput(blob_flip)
        preds_b = self.age_net.forward()[0].astype(np.float32)
        preds = 0.5 * (preds_a + preds_b)
        return preds / max(float(np.sum(preds)), 1e-8)

    def _match_track(self, center_x: float, center_y: float) -> int:
        best_id = -1
        best_dist = self.cfg.track_match_distance_px
        for tid, state in self.tracks.items():
            tx, ty = state["center"]
            dist = float(np.hypot(center_x - tx, center_y - ty))
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        return best_id

    def _prune_tracks(self) -> None:
        stale = [tid for tid, st in self.tracks.items() if int(st["ttl"]) <= 0]
        for tid in stale:
            del self.tracks[tid]

    def annotate_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        for state in self.tracks.values():
            state["ttl"] = int(state["ttl"]) - 1

        bboxes = self.detect_faces(frame)
        results: List[Dict[str, object]] = []
        centers = np.array(self.cfg.age_bucket_centers, dtype=np.float32)

        for x1, y1, x2, y2 in bboxes:
            if (x2 - x1) < self.cfg.min_face_size_px or (y2 - y1) < self.cfg.min_face_size_px:
                continue
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            probs = self.predict_age_probs(face_img)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            tid = self._match_track(cx, cy)

            if tid == -1:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracks[tid] = {
                    "probs": probs,
                    "center": (cx, cy),
                    "ttl": self.cfg.track_ttl_frames,
                }
            else:
                prev_probs = self.tracks[tid]["probs"]
                smoothed = (1.0 - self.cfg.smooth_alpha) * prev_probs + self.cfg.smooth_alpha * probs
                smoothed = smoothed / max(float(np.sum(smoothed)), 1e-8)
                self.tracks[tid]["probs"] = smoothed
                self.tracks[tid]["center"] = (cx, cy)
                self.tracks[tid]["ttl"] = self.cfg.track_ttl_frames

            stable_probs = self.tracks[tid]["probs"]
            idx = int(np.argmax(stable_probs))
            age_bucket = self.cfg.age_buckets[idx]
            confidence = float(stable_probs[idx])
            estimated_age = float(np.dot(stable_probs, centers))
            quality = self._face_quality_score(face_img)
            calibrated_confidence = max(0.0, min(1.0, confidence * (0.75 + 0.25 * quality)))

            result = {
                "track_id": tid,
                "age_bucket": age_bucket,
                "confidence": calibrated_confidence,
                "estimated_age": estimated_age,
                "quality_score": quality,
                "bbox": (x1, y1, x2, y2),
            }
            results.append(result)

            label = f"ID:{tid} {age_bucket} {estimated_age:.1f}y {calibrated_confidence * 100:.0f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_lbl = max(y1, th + 12)
            cv2.rectangle(frame, (x1, y_lbl - th - 10), (x1 + tw + 4, y_lbl + baseline - 8), (0, 220, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1 + 2, y_lbl - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        self._prune_tracks()
        self.analytics.update(results)
        return frame, results

    def _draw_panel(self, frame: np.ndarray, fps: int) -> np.ndarray:
        panel_w = 320
        h, w = frame.shape[:2]
        out = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
        out[:, :w] = frame
        panel = out[:, w:]
        panel[:] = (32, 32, 32)

        lines = [
            "Age Detection Pro",
            f"FPS: {fps}",
            f"Frames: {self.analytics.frame_count}",
            f"Faces(total): {self.analytics.total_face_observations}",
            f"Faces/frame(avg): {self.analytics.avg_faces_per_frame():.2f}",
            f"Unique tracks: {len(self.analytics.unique_tracks)}",
            f"Avg age: {self.analytics.avg_age():.1f}y",
            f"Peak crowd: {self.analytics.max_faces_in_frame}",
        ]
        y = 35
        for i, line in enumerate(lines):
            scale = 0.75 if i == 0 else 0.58
            color = (0, 255, 180) if i == 0 else (220, 220, 220)
            cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            y += 34 if i == 0 else 28

        cv2.putText(panel, "Top age buckets:", (14, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (80, 200, 255), 2)
        y += 38
        for bucket, count in self.analytics.bucket_counter.most_common(4):
            cv2.putText(panel, f"{bucket:>8} : {count}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            y += 24
        cv2.putText(panel, "Keys: q=quit, s=snapshot", (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)
        return out

    def _save_snapshot(self, frame: np.ndarray, snapshot_dir: str) -> str:
        os.makedirs(snapshot_dir, exist_ok=True)
        name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        path = os.path.join(snapshot_dir, name)
        cv2.imwrite(path, frame)
        return path

    def export_report(self, json_path: Optional[str], csv_path: Optional[str]) -> None:
        report = self.analytics.summary_dict()
        if json_path:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[INFO] Report JSON saved: {json_path}")
        if csv_path:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in report.items():
                    if isinstance(v, dict):
                        writer.writerow([k, json.dumps(v)])
                    else:
                        writer.writerow([k, v])
            print(f"[INFO] Report CSV saved: {csv_path}")

    def run_video_source(
        self,
        source: str | int,
        window_title: str = "Age detection",
        save_path: Optional[str] = None,
        snapshot_dir: Optional[str] = None,
        snapshot_interval: float = 0.0,
        show_panel: bool = True,
        headless: bool = False,
        max_frames: int = 0,
    ) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {source!r}")
            return

        writer: Optional[cv2.VideoWriter] = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            target_size = (w + 320, h) if show_panel and not headless else (w, h)
            writer = cv2.VideoWriter(save_path, fourcc, src_fps, target_size)

        print("[INFO] Press 'q' to quit, 's' to save snapshot.")
        prev_t = time.perf_counter()
        last_auto_snapshot = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, _ = self.annotate_frame(frame)
            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            fps = int(1.0 / dt) if dt > 1e-6 else 0

            display = self._draw_panel(annotated, fps) if show_panel and not headless else annotated
            if writer is not None:
                writer.write(display)
            if snapshot_dir and snapshot_interval > 0 and (now - last_auto_snapshot) >= snapshot_interval:
                snap_path = self._save_snapshot(display, snapshot_dir)
                print(f"[INFO] Auto snapshot: {snap_path}")
                last_auto_snapshot = now

            if not headless:
                cv2.imshow(window_title, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s") and snapshot_dir:
                    snap_path = self._save_snapshot(display, snapshot_dir)
                    print(f"[INFO] Snapshot: {snap_path}")

            if max_frames > 0 and self.analytics.frame_count >= max_frames:
                print(f"[INFO] Reached max_frames={max_frames}.")
                break

        cap.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()

    def run_image(self, image_path: str, output_path: Optional[str] = None, show_panel: bool = True) -> None:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Cannot read image: {image_path!r}")
            return
        annotated, detections = self.annotate_frame(frame)
        display = self._draw_panel(annotated, fps=0) if show_panel else annotated
        print("[INFO] Detected:", detections if detections else "no faces")
        if output_path:
            cv2.imwrite(output_path, display)
            print(f"[INFO] Saved: {output_path}")
        cv2.imshow("Age detection (image)", display)
        print("[INFO] Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="High-level age detection from webcam/video/image.")
    p.add_argument("--source", default="0", help="Camera index (0/1/..), image path, or video path.")
    p.add_argument("--weights", default=None, help="Folder containing Caffe model/prototxt files.")
    p.add_argument("--save", default=None, help="Optional path to save processed video.")
    p.add_argument("--out-image", default=None, help="Output path for image mode.")
    p.add_argument("--gpu", action="store_true", help="Use CUDA backend when available.")
    p.add_argument("--conf-threshold", type=float, default=0.5, help="Face confidence threshold.")
    p.add_argument("--min-face", type=int, default=60, help="Minimum face size in pixels.")
    p.add_argument("--smooth-alpha", type=float, default=0.25, help="Temporal smoothing strength.")
    p.add_argument("--track-distance", type=float, default=90.0, help="Track matching distance in pixels.")
    p.add_argument("--snapshot-dir", default=None, help="Directory to save snapshots.")
    p.add_argument("--snapshot-interval", type=float, default=0.0, help="Auto snapshot interval (seconds).")
    p.add_argument("--report-json", default=None, help="Write end-of-session summary JSON.")
    p.add_argument("--report-csv", default=None, help="Write end-of-session summary CSV.")
    p.add_argument("--no-panel", action="store_true", help="Disable right analytics panel.")
    p.add_argument("--headless", action="store_true", help="Run without display window.")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        sys_obj = AgeDetectionSystem(
            weights_dir=args.weights,
            use_gpu=args.gpu,
            confidence_threshold=args.conf_threshold,
            min_face_size=args.min_face,
            smooth_alpha=args.smooth_alpha,
            track_distance=args.track_distance,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[CRITICAL ERROR] {e}")
        sys.exit(1)

    src = args.source
    try:
        if src.isdigit():
            sys_obj.run_video_source(
                int(src),
                save_path=args.save,
                snapshot_dir=args.snapshot_dir,
                snapshot_interval=args.snapshot_interval,
                show_panel=not args.no_panel,
                headless=args.headless,
                max_frames=args.max_frames,
            )
        else:
            if not os.path.isfile(src):
                print(f"[ERROR] Not a file: {src!r}")
                sys.exit(1)
            ext = os.path.splitext(src)[1].lower()
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
            if ext in image_exts:
                sys_obj.run_image(src, output_path=args.out_image, show_panel=not args.no_panel)
            else:
                sys_obj.run_video_source(
                    src,
                    save_path=args.save,
                    snapshot_dir=args.snapshot_dir,
                    snapshot_interval=args.snapshot_interval,
                    show_panel=not args.no_panel,
                    headless=args.headless,
                    max_frames=args.max_frames,
                )
    finally:
        sys_obj.export_report(args.report_json, args.report_csv)


if __name__ == "__main__":
    main()
