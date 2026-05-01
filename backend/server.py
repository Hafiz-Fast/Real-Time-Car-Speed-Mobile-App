import asyncio
import base64
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
MODEL_PATH = ROOT.parent / "yolov8n.pt"

VEHICLE_CLASSES = [2, 3, 5, 7]


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    # Fallback: a very wide zone that covers the whole frame
    # (will be replaced once calibrated from mobile)
    return {
        "real_width_m": 7.0,
        "real_height_m": 60.0,
        "source_points": [
            [0, 0],
            [1280, 0],
            [1280, 720],
            [0, 720],
        ],
    }


def build_calibration(config: Dict[str, Any]) -> Dict[str, Any]:
    real_width = float(config.get("real_width_m", 7.0))
    real_height = float(config.get("real_height_m", 60.0))

    source_points = np.float32(config.get("source_points"))
    dest_points = np.float32(
        [
            [0, 0],
            [real_width, 0],
            [real_width, real_height],
            [0, real_height],
        ]
    )

    homography, _ = cv2.findHomography(source_points, dest_points)
    zone_polygon = source_points.astype(np.int32)

    return {
        "real_width": real_width,
        "real_height": real_height,
        "source_points": source_points,
        "dest_points": dest_points,
        "homography": homography,
        "zone_polygon": zone_polygon,
    }


# Mutable global calibration (replaced live when mobile calibrates)
CONFIG = load_config()
CALIBRATION = build_calibration(CONFIG)
MODEL = YOLO(str(MODEL_PATH))

app = FastAPI()


def pixel_to_world(pixel_point: Any, homography: np.ndarray) -> np.ndarray:
    point = np.array([[pixel_point]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(point, homography)
    return world_point[0][0]


def point_in_zone(px: int, py: int, zone_polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(zone_polygon, (float(px), float(py)), False) >= 0


def resize_for_inference(frame: np.ndarray, width: int = 640) -> np.ndarray:
    height, frame_width = frame.shape[:2]
    scale = width / frame_width
    return cv2.resize(frame, (width, int(height * scale)))


def decode_base64_image(image_base64: str) -> np.ndarray:
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image")
    return frame


class SpeedEstimator:
    def __init__(self, model: YOLO) -> None:
        self.model = model
        self.tracker = DeepSort(
            max_age=30,          # keep track alive for 30 missed frames (~2 s at 15 FPS)
            n_init=2,            # confirm a track after only 2 detections (was 3)
            nms_max_overlap=0.7, # allow more overlapping boxes through
        )
        # history stores (wall_time_seconds, world_x, world_y)
        self.track_history: Dict[Any, deque] = defaultdict(lambda: deque(maxlen=30))
        self.track_speeds: Dict[Any, float] = defaultdict(float)
        self.frame_count = 0
        self.skip_frames = 2
        self.min_positions = 4

    def process_frame(
        self, frame: np.ndarray, calibration: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            return {
                "detections": [],
                "frame_size": [frame.shape[1], frame.shape[0]],
                "skipped": True,
            }

        small_frame = resize_for_inference(frame, width=640)
        height, width = frame.shape[:2]
        small_height, small_width = small_frame.shape[:2]
        scale_x = width / small_width
        scale_y = height / small_height

        results = self.model(small_frame, verbose=False, imgsz=416)[0]
        raw_detections: List[Any] = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id in VEHICLE_CLASSES and conf > 0.4:  # lowered threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                raw_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))

        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        detections: List[Dict[str, Any]] = []
        now = time.time()

        homography = calibration["homography"]
        zone_polygon = calibration["zone_polygon"]

        for track in tracks:
            if not track.is_confirmed():
                continue
            if track.time_since_update > 3:
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx = (x1 + x2) // 2
            cy = y2  # bottom-center of bounding box

            if point_in_zone(cx, cy, zone_polygon):
                world_pos = pixel_to_world((cx, cy), homography)
                # Store REAL wall-clock time instead of frame count
                self.track_history[track_id].append(
                    (now, world_pos[0], world_pos[1])
                )

                history = self.track_history[track_id]
                if len(history) >= self.min_positions:
                    oldest = history[0]
                    newest = history[-1]

                    dx = newest[1] - oldest[1]
                    dy = newest[2] - oldest[2]
                    dist_meters = float(np.sqrt(dx**2 + dy**2))

                    # Real elapsed time in seconds
                    time_elapsed = newest[0] - oldest[0]

                    if time_elapsed > 0.1:  # at least 100ms of history
                        speed_kmh = (dist_meters / time_elapsed) * 3.6
                        if 0 < speed_kmh < 250:
                            prev = self.track_speeds[track_id]
                            alpha = 0.5  # smoothing
                            if prev == 0:
                                self.track_speeds[track_id] = speed_kmh
                            else:
                                self.track_speeds[track_id] = (
                                    alpha * prev + (1 - alpha) * speed_kmh
                                )

            speed = self.track_speeds.get(track_id, 0.0)

            if speed >= 100:
                color = [0, 0, 255]   # red — fast
            elif speed >= 60:
                color = [0, 165, 255]  # orange — medium
            elif speed > 0:
                color = [0, 255, 0]   # green — slow
            else:
                color = [200, 200, 200]  # gray — no speed yet

            label = f"ID:{track_id} {speed:.1f} km/h" if speed > 0 else f"ID:{track_id}"

            detections.append(
                {
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "speed_kmh": float(speed),
                    "label": label,
                    "color": color,
                }
            )

        return {
            "detections": detections,
            "frame_size": [width, height],
            "skipped": False,
        }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/calibration")
async def get_calibration() -> Dict[str, Any]:
    """Return the current calibration so the mobile app can display it."""
    return {
        "real_width_m": CONFIG.get("real_width_m", 7.0),
        "real_height_m": CONFIG.get("real_height_m", 60.0),
        "source_points": CONFIG.get("source_points", []),
        "is_calibrated": CONFIG_PATH.exists(),
    }


@app.websocket("/ws/calibrate")
async def calibrate_endpoint(websocket: WebSocket) -> None:
    """
    Receives a JSON payload from mobile:
    {
        "image_base64": "<base64 jpeg>",
        "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],  // in TL,TR,BR,BL order
        "real_width_m": 7.0,
        "real_height_m": 60.0
    }
    Saves config.json, reloads calibration, responds with confirmation.
    """
    global CONFIG, CALIBRATION

    await websocket.accept()
    try:
        message = await websocket.receive_text()
        payload = json.loads(message)

        points = payload.get("points")
        real_width = float(payload.get("real_width_m", 7.0))
        real_height = float(payload.get("real_height_m", 60.0))

        if not points or len(points) != 4:
            await websocket.send_text(
                json.dumps({"success": False, "error": "Need exactly 4 points"})
            )
            return

        # Validate points are within frame bounds (basic sanity check)
        for p in points:
            if len(p) != 2:
                await websocket.send_text(
                    json.dumps({"success": False, "error": "Each point must be [x, y]"})
                )
                return

        new_config = {
            "real_width_m": real_width,
            "real_height_m": real_height,
            "source_points": points,
        }

        CONFIG_PATH.write_text(json.dumps(new_config, indent=2), encoding="utf-8")
        CONFIG = new_config
        CALIBRATION = build_calibration(new_config)

        await websocket.send_text(
            json.dumps(
                {
                    "success": True,
                    "message": "Calibration saved and applied",
                    "source_points": points,
                    "real_width_m": real_width,
                    "real_height_m": real_height,
                }
            )
        )
    except Exception as exc:
        await websocket.send_text(json.dumps({"success": False, "error": str(exc)}))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    estimator = SpeedEstimator(MODEL)
    loop = asyncio.get_running_loop()

    try:
        while True:
            message = await websocket.receive_text()
            payload = json.loads(message)
            frame_id = payload.get("frame_id")
            image_base64 = payload.get("image_base64")

            if not image_base64:
                await websocket.send_text(
                    json.dumps({"frame_id": frame_id, "error": "Missing image"})
                )
                continue

            frame = decode_base64_image(image_base64)

            # Use the current global calibration (updated live after /ws/calibrate)
            cal_snapshot = CALIBRATION

            result = await loop.run_in_executor(
                None, estimator.process_frame, frame, cal_snapshot
            )

            response = {
                "frame_id": frame_id,
                "detections": result["detections"],
                "frame_size": result["frame_size"],
                "skipped": result["skipped"],
            }
            await websocket.send_text(json.dumps(response))
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_text(json.dumps({"error": str(exc)}))