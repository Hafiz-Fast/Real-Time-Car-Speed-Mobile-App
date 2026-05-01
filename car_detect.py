from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from collections import defaultdict, deque

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=15)  # Fix 3: reduced from 50 to prevent ghost tracks
VEHICLE_CLASSES = [2, 3, 5, 7]

SOURCE_POINTS = np.float32([
    [209, 276],  # top-left
    [293, 276],  # top-right
    [428, 663],  # bottom-right
    [7, 529],  # bottom-left
])

REAL_WIDTH  = 7.0
REAL_HEIGHT = 60.0

DEST_POINTS = np.float32([
    [0, 0],
    [REAL_WIDTH, 0],
    [REAL_WIDTH, REAL_HEIGHT],
    [0, REAL_HEIGHT],
])

H, _ = cv2.findHomography(SOURCE_POINTS, DEST_POINTS)
ZONE_POLYGON = SOURCE_POINTS.astype(np.int32)

def pixel_to_world(pixel_point):
    pt = np.array([[pixel_point]], dtype=np.float32)
    world_pt = cv2.perspectiveTransform(pt, H)
    return world_pt[0][0]

def point_in_zone(px, py):
    return cv2.pointPolygonTest(ZONE_POLYGON, (float(px), float(py)), False) >= 0

def resize_for_inference(frame, width=640):
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))

def fit_to_screen(frame, screen_width=1280, screen_height=720):
    h, w = frame.shape[:2]
    scale = min(screen_width / w, screen_height / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

# Speed tracking state
track_history = defaultdict(lambda: deque(maxlen=25))
track_speeds  = defaultdict(float)

cap = cv2.VideoCapture("test4.mp4")
FPS = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {FPS}")

frame_count = 0
SKIP_FRAMES = 2
MIN_POSITIONS_FOR_SPEED = 4

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        continue

    small_frame = resize_for_inference(frame, width=640)
    h_orig, w_orig = frame.shape[:2]
    h_small, w_small = small_frame.shape[:2]
    scale_x = w_orig / w_small
    scale_y = h_orig / h_small

    results = model(small_frame, verbose=False, imgsz=416)[0]

    raw_detections = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        if class_id in VEHICLE_CLASSES and conf > 0.5:  # Fix 2: raised from 0.4 to 0.5
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = int(x1*scale_x), int(y1*scale_y)
            x2, y2 = int(x2*scale_x), int(y2*scale_y)
            raw_detections.append(([x1, y1, x2-x1, y2-y1], conf, class_id))

    tracks = tracker.update_tracks(raw_detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        # Fix 1: Skip coasting/drifting tracks — only draw if matched to a real detection this frame
        if track.time_since_update > 1:
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx = (x1 + x2) // 2
        cy = y2  # bottom center

        in_zone = point_in_zone(cx, cy)

        if in_zone:
            world_pos = pixel_to_world((cx, cy))
            track_history[track_id].append((frame_count, world_pos[0], world_pos[1]))

            history = track_history[track_id]

            if len(history) >= MIN_POSITIONS_FOR_SPEED:
                oldest = history[0]
                newest = history[-1]

                dx = newest[1] - oldest[1]
                dy = newest[2] - oldest[2]
                dist_meters = np.sqrt(dx**2 + dy**2)

                frames_elapsed = newest[0] - oldest[0]
                time_elapsed = frames_elapsed / FPS

                if time_elapsed > 0:
                    speed_kmh = (dist_meters / time_elapsed) * 3.6

                    if 0 < speed_kmh < 250:
                        prev = track_speeds[track_id]
                        if prev == 0:
                            track_speeds[track_id] = speed_kmh
                        else:
                            track_speeds[track_id] = 0.4 * prev + 0.6 * speed_kmh

        speed = track_speeds.get(track_id, 0)

        if speed > 0:
            color = (0, 255, 0) if speed < 60 else (0, 165, 255) if speed < 100 else (0, 0, 255)
            label = f"ID:{track_id}  {speed:.1f} km/h"
        else:
            color = (0, 255, 0)
            label = f"ID:{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Draw calibration zone
    pts = SOURCE_POINTS.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    display = fit_to_screen(frame)
    cv2.imshow("Speed Estimation", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()