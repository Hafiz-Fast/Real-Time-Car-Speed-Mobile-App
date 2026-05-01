import asyncio
import base64
import json
import os
import tempfile
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
MODEL_PATH = ROOT.parent / "yolov8n.pt"

# Directory to store processed videos temporarily
OUTPUT_DIR = ROOT / "processed_videos"
OUTPUT_DIR.mkdir(exist_ok=True)

VEHICLE_CLASSES = [2, 3, 5, 7]

# ─── Config / Calibration ────────────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "real_width_m": 7.0,
        "real_height_m": 60.0,
        "source_points": [[0, 0], [1280, 0], [1280, 720], [0, 720]],
    }


def build_calibration(config: Dict[str, Any]) -> Dict[str, Any]:
    real_width  = float(config.get("real_width_m",  7.0))
    real_height = float(config.get("real_height_m", 60.0))

    source_points = np.float32(config.get("source_points"))
    dest_points   = np.float32([
        [0,           0],
        [real_width,  0],
        [real_width,  real_height],
        [0,           real_height],
    ])

    homography, _  = cv2.findHomography(source_points, dest_points)
    zone_polygon   = source_points.astype(np.int32)

    return {
        "real_width":         real_width,
        "real_height":        real_height,
        "source_points":      source_points,
        "dest_points":        dest_points,
        "homography":         homography,
        "zone_polygon":       zone_polygon,
        "calibrated_width":   float(config.get("frame_width",  1280)),
        "calibrated_height":  float(config.get("frame_height", 720)),
    }


CONFIG      = load_config()
CALIBRATION = build_calibration(CONFIG)
MODEL       = YOLO(str(MODEL_PATH))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track processing jobs: job_id -> {"status": ..., "progress": ..., "output_path": ...}
JOBS: Dict[str, Dict[str, Any]] = {}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def pixel_to_world(pixel_point: Any, homography: np.ndarray) -> np.ndarray:
    point       = np.array([[pixel_point]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(point, homography)
    return world_point[0][0]


def point_in_zone(px: int, py: int, zone_polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(zone_polygon, (float(px), float(py)), False) >= 0


def rescale_calibration_to_frame(
    calibration: Dict[str, Any], frame_w: int, frame_h: int
) -> Dict[str, Any]:
    cal_w = calibration.get("calibrated_width",  frame_w)
    cal_h = calibration.get("calibrated_height", frame_h)

    if abs(cal_w - frame_w) < 2 and abs(cal_h - frame_h) < 2:
        return calibration

    sx = frame_w / cal_w
    sy = frame_h / cal_h

    original_src = np.float32(calibration["source_points"])
    new_src      = original_src * np.float32([sx, sy])
    dest_points  = calibration["dest_points"]
    homography, _ = cv2.findHomography(new_src, dest_points)
    zone_polygon  = new_src.astype(np.int32)

    return {
        **calibration,
        "source_points": new_src,
        "homography":    homography,
        "zone_polygon":  zone_polygon,
    }


def decode_base64_image(image_base64: str) -> np.ndarray:
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes  = base64.b64decode(image_base64)
    image_array  = np.frombuffer(image_bytes, np.uint8)
    frame        = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image")
    return frame


def draw_zone_overlay(frame: np.ndarray, zone_polygon: np.ndarray) -> np.ndarray:
    """Draw semi-transparent calibrated zone polygon on frame."""
    overlay = frame.copy()
    pts     = zone_polygon.reshape((-1, 1, 2))

    # Fill with semi-transparent blue
    cv2.fillPoly(overlay, [pts], (255, 180, 0))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Draw solid border
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 200, 0), thickness=3)

    # Corner labels
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(255, 100, 100), (100, 255, 220), (255, 220, 50), (100, 255, 120)]
    for i, (pt, lbl, col) in enumerate(zip(zone_polygon, labels, colors)):
        cx, cy = int(pt[0]), int(pt[1])
        cv2.circle(frame, (cx, cy), 10, col, -1)
        cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
        cv2.putText(frame, lbl, (cx + 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    return frame


def draw_detection(
    frame: np.ndarray,
    det: Dict[str, Any],
) -> np.ndarray:
    x1, y1, x2, y2 = det["bbox"]
    b, g, r         = det["color"]
    bgr             = (b, g, r)
    speed           = det["speed_kmh"]
    label           = det["label"]

    # Bounding box — thick and visible
    thickness = 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thickness)

    # Label background
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thick = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
    label_y1   = max(y1 - th - baseline - 8, 0)
    label_y2   = max(y1, th + baseline + 8)

    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 8, label_y2), bgr, -1)
    cv2.putText(frame, label, (x1 + 4, label_y2 - baseline - 2),
                font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    # Speed badge at bottom-center of box if speeding
    if speed >= 100:
        badge   = f"⚠ {speed:.0f}"
        bx      = (x1 + x2) // 2
        by      = y2 + 20
        (bw, bh), _ = cv2.getTextSize(badge, font, 0.6, 2)
        cv2.rectangle(frame, (bx - bw // 2 - 4, by - bh - 4),
                      (bx + bw // 2 + 4, by + 4), (0, 0, 220), -1)
        cv2.putText(frame, badge, (bx - bw // 2, by),
                    font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_hud(frame: np.ndarray, frame_num: int, total_frames: int,
             fps: float, n_vehicles: int, timestamp_s: float) -> np.ndarray:
    """Draw HUD overlay: frame counter, vehicle count, timestamp."""
    h, w = frame.shape[:2]
    bar_h = 42
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 20), -1)

    minutes = int(timestamp_s // 60)
    seconds = timestamp_s % 60
    ts_str  = f"{minutes:02d}:{seconds:05.2f}"
    pct     = f"{frame_num}/{total_frames}"

    texts = [
        (f"⏱ {ts_str}",        20),
        (f"🚗 {n_vehicles} vehicles", 220),
        (f"Frame {pct}",        w - 200),
    ]
    for txt, x in texts:
        cv2.putText(frame, txt, (x, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

    # Progress bar at very bottom
    progress = frame_num / max(total_frames, 1)
    bar_y    = h - 8
    cv2.rectangle(frame, (0, bar_y), (w, h), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, bar_y), (int(w * progress), h), (26, 188, 156), -1)

    return frame


# ─── Speed Estimator ─────────────────────────────────────────────────────────

class SpeedEstimator:
    def __init__(self, model: YOLO) -> None:
        self.model         = model
        self.tracker       = DeepSort(max_age=30, n_init=2, nms_max_overlap=0.7)
        self.track_history: Dict[Any, deque]  = defaultdict(lambda: deque(maxlen=60))
        self.track_speeds:  Dict[Any, float]  = defaultdict(float)
        self.min_positions = 3

    def process_frame(
        self,
        frame: np.ndarray,
        calibration: Dict[str, Any],
        timestamp: float,
    ) -> Dict[str, Any]:
        frame_h, frame_w = frame.shape[:2]
        cal  = rescale_calibration_to_frame(calibration, frame_w, frame_h)

        # Inference on resized frame
        infer_w       = 640
        scale         = infer_w / frame_w
        small_frame   = cv2.resize(frame, (infer_w, int(frame_h * scale)))
        sh, sw        = small_frame.shape[:2]
        scale_x, scale_y = frame_w / sw, frame_h / sh

        results       = self.model(small_frame, verbose=False, imgsz=640)[0]
        raw_detections: List[Any] = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf     = float(box.conf[0])
            if class_id in VEHICLE_CLASSES and conf > 0.35:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
                raw_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))

        tracks     = self.tracker.update_tracks(raw_detections, frame=frame)
        detections: List[Dict[str, Any]] = []
        homography  = cal["homography"]
        zone_polygon = cal["zone_polygon"]

        for track in tracks:
            if not track.is_confirmed():
                continue
            if track.time_since_update > 3:
                continue

            track_id   = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy     = (x1 + x2) // 2, y2   # bottom-center

            in_zone    = point_in_zone(cx, cy, zone_polygon)

            if in_zone:
                world_pos = pixel_to_world((cx, cy), homography)
                self.track_history[track_id].append(
                    (timestamp, world_pos[0], world_pos[1])
                )

                history = self.track_history[track_id]
                if len(history) >= self.min_positions:
                    oldest     = history[0]
                    newest     = history[-1]
                    dx         = newest[1] - oldest[1]
                    dy         = newest[2] - oldest[2]
                    dist_m     = float(np.sqrt(dx**2 + dy**2))
                    dt         = newest[0] - oldest[0]

                    if dt > 0.1:
                        speed_kmh = (dist_m / dt) * 3.6
                        if 1 < speed_kmh < 250:
                            prev  = self.track_speeds[track_id]
                            alpha = 0.4
                            self.track_speeds[track_id] = (
                                speed_kmh if prev == 0
                                else alpha * prev + (1 - alpha) * speed_kmh
                            )

            speed = self.track_speeds.get(track_id, 0.0)

            if speed >= 100:    color = [0, 0, 255]
            elif speed >= 60:   color = [0, 165, 255]
            elif speed > 0:     color = [0, 255, 0]
            else:               color = [180, 180, 180]

            label = f"ID:{track_id}  {speed:.1f} km/h" if speed > 0 else f"ID:{track_id}"

            detections.append({
                "id":        track_id,
                "bbox":      [x1, y1, x2, y2],
                "speed_kmh": float(speed),
                "label":     label,
                "color":     color,
                "in_zone":   in_zone,
            })

        return {"detections": detections, "frame_size": [frame_w, frame_h]}


# ─── Video Processing Job ────────────────────────────────────────────────────

def process_video_job(
    job_id:      str,
    video_path:  str,
    calibration: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Runs in a thread pool. Reads every frame, runs YOLO+DeepSort,
    draws zone + bounding boxes, writes annotated video to output_path.
    Updates JOBS[job_id] with progress.
    """
    try:
        JOBS[job_id]["status"] = "processing"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        fps_in      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output codec — H.264 for broad mobile support
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps_in, (frame_w, frame_h))

        estimator = SpeedEstimator(MODEL)
        cal       = rescale_calibration_to_frame(calibration, frame_w, frame_h)

        frame_num  = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            timestamp  = frame_num / fps_in

            result = estimator.process_frame(frame, cal, timestamp)

            # Draw calibration zone first (below boxes)
            frame = draw_zone_overlay(frame, cal["zone_polygon"])

            # Draw each detection
            for det in result["detections"]:
                frame = draw_detection(frame, det)

            # HUD
            frame = draw_hud(
                frame, frame_num, total_frames,
                fps_in, len(result["detections"]), timestamp
            )

            out.write(frame)

            # Update progress
            progress = round(frame_num / max(total_frames, 1) * 100, 1)
            JOBS[job_id]["progress"] = progress

        cap.release()
        out.release()

        JOBS[job_id]["status"]      = "done"
        JOBS[job_id]["progress"]    = 100.0
        JOBS[job_id]["output_path"] = output_path

    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"]  = str(exc)
    finally:
        # Clean up the input temp file
        try:
            os.remove(video_path)
        except OSError:
            pass


# ─── REST / WS Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/calibration")
async def get_calibration() -> Dict[str, Any]:
    return {
        "real_width_m":  CONFIG.get("real_width_m",  7.0),
        "real_height_m": CONFIG.get("real_height_m", 60.0),
        "source_points": CONFIG.get("source_points", []),
        "is_calibrated": CONFIG_PATH.exists(),
    }


@app.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    calibration_json: str = Form(...),   # JSON string of calibration config
) -> Dict[str, Any]:
    """
    Accept a video file + calibration JSON from mobile.
    Kick off background processing, return job_id immediately.
    """
    global CONFIG, CALIBRATION

    # Parse calibration override if provided
    try:
        cal_config = json.loads(calibration_json)
    except Exception:
        cal_config = CONFIG

    calibration = build_calibration(cal_config)

    # Save upload to temp file
    suffix     = Path(video.filename or "video.mp4").suffix or ".mp4"
    tmp_input  = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(OUTPUT_DIR))
    contents   = await video.read()
    tmp_input.write(contents)
    tmp_input.close()

    job_id      = str(uuid.uuid4())
    output_path = str(OUTPUT_DIR / f"{job_id}_processed.mp4")

    JOBS[job_id] = {
        "status":      "queued",
        "progress":    0.0,
        "output_path": None,
        "error":       None,
    }

    # Run processing in background (CPU-bound sync function)
    background_tasks.add_task(
        _run_sync_job, job_id, tmp_input.name, calibration, output_path
    )

    return {"job_id": job_id, "status": "queued"}


def _run_sync_job(
    job_id: str, video_path: str,
    calibration: Dict[str, Any], output_path: str
) -> None:
    """Thin sync wrapper so FastAPI BackgroundTasks can call process_video_job."""
    process_video_job(job_id, video_path, calibration, output_path)


@app.get("/job/{job_id}")
async def job_status(job_id: str) -> Dict[str, Any]:
    """Poll this endpoint to check processing progress."""
    if job_id not in JOBS:
        return {"status": "not_found"}
    job = JOBS[job_id]
    return {
        "status":   job["status"],
        "progress": job["progress"],
        "error":    job.get("error"),
    }


@app.get("/job/{job_id}/download")
async def download_result(job_id: str):
    """Download the processed video once status == 'done'."""
    if job_id not in JOBS:
        return {"error": "not found"}
    job = JOBS[job_id]
    if job["status"] != "done":
        return {"error": "not ready", "status": job["status"]}
    path = job["output_path"]
    if not path or not Path(path).exists():
        return {"error": "file missing"}
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"processed_{job_id[:8]}.mp4",
        headers={"Content-Disposition": f'attachment; filename="processed_{job_id[:8]}.mp4"'},
    )


@app.delete("/job/{job_id}")
async def delete_job(job_id: str) -> Dict[str, str]:
    """Cleanup after mobile has downloaded the result."""
    if job_id in JOBS:
        path = JOBS[job_id].get("output_path")
        if path:
            try:
                os.remove(path)
            except OSError:
                pass
        del JOBS[job_id]
    return {"deleted": job_id}


# ─── WebSocket: Calibrate ─────────────────────────────────────────────────────

@app.websocket("/ws/calibrate")
async def calibrate_endpoint(websocket: WebSocket) -> None:
    global CONFIG, CALIBRATION
    await websocket.accept()
    try:
        message = await websocket.receive_text()
        payload = json.loads(message)

        points       = payload.get("points")
        real_width   = float(payload.get("real_width_m",  7.0))
        real_height  = float(payload.get("real_height_m", 60.0))
        frame_width  = int(payload.get("frame_width",  1280))
        frame_height = int(payload.get("frame_height", 720))

        if not points or len(points) != 4:
            await websocket.send_text(
                json.dumps({"success": False, "error": "Need exactly 4 points"})
            )
            return

        new_config = {
            "real_width_m":  real_width,
            "real_height_m": real_height,
            "source_points": points,
            "frame_width":   frame_width,
            "frame_height":  frame_height,
        }
        CONFIG_PATH.write_text(json.dumps(new_config, indent=2), encoding="utf-8")
        CONFIG      = new_config
        CALIBRATION = build_calibration(new_config)

        await websocket.send_text(json.dumps({
            "success":       True,
            "message":       "Calibration saved and applied",
            "source_points": points,
            "real_width_m":  real_width,
            "real_height_m": real_height,
            "frame_width":   frame_width,
            "frame_height":  frame_height,
        }))
    except Exception as exc:
        await websocket.send_text(json.dumps({"success": False, "error": str(exc)}))


# ─── WebSocket: Live (kept for optional use) ─────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    estimator = SpeedEstimator(MODEL)
    loop      = asyncio.get_running_loop()
    frame_num = 0

    try:
        while True:
            message    = await websocket.receive_text()
            payload    = json.loads(message)
            frame_id   = payload.get("frame_id")
            image_b64  = payload.get("image_base64")

            if not image_b64:
                await websocket.send_text(
                    json.dumps({"frame_id": frame_id, "error": "Missing image"})
                )
                continue

            frame      = decode_base64_image(image_b64)
            frame_num += 1
            timestamp  = frame_num / 10.0   # assume ~10 fps from mobile

            result = await loop.run_in_executor(
                None, estimator.process_frame, frame, CALIBRATION, timestamp
            )
            await websocket.send_text(json.dumps({
                "frame_id":   frame_id,
                "detections": result["detections"],
                "frame_size": result["frame_size"],
            }))
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_text(json.dumps({"error": str(exc)}))