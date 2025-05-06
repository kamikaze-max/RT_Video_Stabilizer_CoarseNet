# camera_stream.py

import cv2

def run_webcam():
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("✅ Webcam opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Resize for testing if needed
        frame_resized = cv2.resize(frame, (640, 480))
        cv2.imshow("Webcam Feed", frame_resized)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

