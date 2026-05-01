# Real-Time-Car-Speed-Mobile-App
This project aims to implement a real time Car detection and its speed calculation using Computer Vision techniques and tools. The project requires a mobile app so that mobile camera can be used and can display real time car speed moving on a road.

## Backend (FastAPI + WebSocket)

The backend accepts base64 JPEG frames over WebSocket, runs detection + tracking, and returns bounding boxes with speed labels.

### Install

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Calibration config

Update calibration points in [backend/config.json](backend/config.json). Use the values printed by calibrate.py.

## Mobile (Expo)

The Expo app streams frames to the backend and overlays the returned boxes and speed labels.

See setup steps in [mobile-app/README.md](mobile-app/README.md).
