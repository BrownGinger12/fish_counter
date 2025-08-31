import cv2
from ultralytics import YOLO
import serial
import time

# Load YOLO model
model = YOLO("my_model.pt")

# Open Arduino serial connection
arduino = serial.Serial("COM6", 9600, timeout=1)  # Change COM port if needed
time.sleep(2)

# Mode variables
mode = None   # <-- no mode set by default
target_num = None
fish_count = 0
counted_ids = set()

def read_from_arduino():
    """Reads mode/reset commands from Arduino."""
    global mode, target_num, fish_count, counted_ids
    if arduino.in_waiting > 0:
        line = arduino.readline().decode(errors="ignore").strip()
        if not line:
            return

        if line.startswith("MODE:EXACT"):
            parts = line.split(",")
            if len(parts) == 2 and "NUM:" in parts[1]:
                mode = "EXACT"
                target_num = int(parts[1].split(":")[1])
                fish_count = 0
                counted_ids.clear()
                print(f"🔹 Exact Mode set, Target: {target_num}")

        elif line.startswith("MODE:COUNT"):
            mode = "COUNT"
            target_num = None
            fish_count = 0
            counted_ids.clear()
            print("🔹 Count Mode set")

        elif line.startswith("RESET:MENU"):
            mode = None
            target_num = None
            fish_count = 0
            counted_ids.clear()
            print("🔄 Back to MENU (Select Mode)")

# Video input
cap = cv2.VideoCapture("test3.mp4")

line_x = 900

while True:
    ret, frame = cap.read()
    if not ret:
        break

    read_from_arduino()  # check commands from Arduino

    # Show waiting message if no mode is set
    if mode is None:
        cv2.putText(frame, "Waiting for MODE from Arduino...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Fish Counter", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        continue  # Skip counting until mode is set

    # Run YOLO tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            track_id = int(box.id.cpu().numpy())
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().astype(int)
            cx, cy = int((xmin + xmax) / 2), int((ymin + ymax) / 2)

            # Draw
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Counting logic
            if track_id not in counted_ids:
                if cx > line_x:
                    fish_count += 1
                    counted_ids.add(track_id)
                    arduino.write(f"COUNT:{fish_count}\n".encode())  # send count to Arduino

    # Draw line + count
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)
    cv2.putText(frame, f'Fish crossed: {fish_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Fish Counter", frame)

    # Stop if target reached
    if mode == "EXACT" and target_num is not None and fish_count >= target_num:
        print("✅ Target reached! Stopping count.")
        mode = None
        target_num = None
        fish_count = 0
        counted_ids.clear()

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
