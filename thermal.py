import requests
import numpy as np
import cv2
import time

ESP32_IP = "http://10.62.16.77"  # <-- change to your ESP32 IP
WIDTH, HEIGHT = 80, 60
FPS = 8
DELAY = 1.0 / FPS

while True:
    start = time.time()
    try:
        r = requests.get(f"{ESP32_IP}/frame8.raw", timeout=0.5)
        if r.status_code != 200:
            print("Bad response:", r.status_code)
            continue

        data = np.frombuffer(r.content, dtype=np.uint8)
        if data.size != WIDTH * HEIGHT:
            print("Bad frame size")
            continue

        frame8 = data.reshape((HEIGHT, WIDTH))

        # Apply a colormap (e.g., HOT, JET, or INFERNO)
        frame_color = cv2.applyColorMap(frame8, cv2.COLORMAP_JET)

        # Resize for viewing
        frame_up = cv2.resize(frame_color, (640, 480), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Lepton Stream", frame_up)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

        # Maintain FPS
        elapsed = time.time() - start
        if elapsed < DELAY:
            time.sleep(DELAY - elapsed)

    except Exception as e:
        print("Error:", e)

cv2.destroyAllWindows()
