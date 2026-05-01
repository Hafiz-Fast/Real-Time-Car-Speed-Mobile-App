# calibrate.py - improved version with guidance
import cv2
import numpy as np

points = []
ORDER_LABELS = ["1: TOP-LEFT", "2: TOP-RIGHT", "3: BOTTOM-RIGHT", "4: BOTTOM-LEFT"]

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        label = ORDER_LABELS[len(points)-1]
        print(f"{label}: ({x}, {y})")
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(img, label, (x+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Calibrate - Click in order: TL, TR, BR, BL", img)

        if len(points) == 4:
            # Draw the trapezoid to verify it looks right
            pts = np.array(points, np.int32).reshape((-1,1,2))
            cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
            cv2.imshow("Calibrate - Click in order: TL, TR, BR, BL", img)
            print("\n── Copy these into step3_speed.py ──")
            print(f"SOURCE_POINTS = np.float32({points})")

cap = cv2.VideoCapture("test4.mp4")
ret, img = cap.read()
cap.release()

print("Click 4 road points in this ORDER: Top-Left, Top-Right, Bottom-Right, Bottom-Left")

window_name = "Calibrate - Click in order: TL, TR, BR, BL"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

cv2.imshow(window_name, img)
cv2.setMouseCallback(window_name, click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()