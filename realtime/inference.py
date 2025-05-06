# inference.py

import time
import cv2
import torch
import numpy as np
from models.coarse_net import CoarseNet
from models.fine_net import FineNet
from utils.flow import compute_optical_flow


def warp_image_with_flow(image, flow):
    """Fast warp using dense flow via OpenCV remap (vectorized)."""
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def warp_image_global(image, transform_params):
    """Apply global 2D rigid transformation to the image."""
    h, w = image.shape[:2]
    theta, tx, ty = transform_params

    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    transform_matrix = np.array([
        [cos_a, -sin_a, tx * w],
        [sin_a,  cos_a, ty * h]
    ], dtype=np.float32)

    return cv2.warpAffine(image, transform_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
 

def run_real_time_stabilization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coarse_net = CoarseNet().to(device).eval()
    fine_net = FineNet().to(device).eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        return

    prev_frame = None
    print("🎥 Running real-time stabilization with recording... Press 'q' to quit.")

    # === Video writer setup ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_size = (256 * 3, 256)  # Side-by-side width
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = cv2.VideoWriter(f"output_stabilized_{timestamp}.mp4", fourcc, 20.0, output_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (256, 256))

        if prev_frame is not None:
            flow_np = compute_optical_flow(prev_frame, frame_resized)
            flow_tensor = torch.from_numpy(flow_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

            with torch.no_grad():
                # CoarseNet → Use fixed params or model output
                # coarse_params = coarse_net(flow_tensor).cpu().numpy()[0]
                coarse_params = np.array([0.0, 0.0, 0.0])

                coarse_stabilized = warp_image_global(frame_resized, coarse_params)

                # FineNet
                smoothed_flow_tensor = fine_net(flow_tensor)
                smoothed_flow_np = smoothed_flow_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                final_stabilized = warp_image_with_flow(coarse_stabilized, smoothed_flow_np)

            # Stack views
            combined = np.hstack((frame_resized, coarse_stabilized, final_stabilized))

            # Add labels (optional polish)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, 'Original', (10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(combined, 'Coarse', (266, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(combined, 'Final', (526, 20), font, 0.5, (255, 255, 255), 1)

            # Show and record
            cv2.imshow("Original | Coarse | Coarse+Fine", combined)
            out.write(combined)
        else:
            blank = np.hstack([frame_resized]*3)
            cv2.imshow("Original | Coarse | Coarse+Fine", blank)
            out.write(blank)

        prev_frame = frame_resized.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

