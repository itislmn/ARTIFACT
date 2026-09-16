"""
Inspect and plot the .npz saved by step1_uart_pointcloud.py

    python inspect_capture.py uart_capture.npz
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "uart_capture.npz"
data = np.load(path, allow_pickle=True)

frame_ids = data["frame_ids"]
num_obj = data["num_obj"]
points = data["points"]  # object array, one (N,4) array per frame with a point

print(f"File: {path}")
print(f"Frames captured : {len(frame_ids)}")
print(f"Frame id range  : {frame_ids.min()} - {frame_ids.max()}")
print(f"Objects/frame   : min={num_obj.min()} max={num_obj.max()} mean={num_obj.mean():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. object count over time - proves the sensor is seeing something changing
axes[0].plot(frame_ids, num_obj)
axes[0].set_xlabel("Frame number")
axes[0].set_ylabel("Detected objects")
axes[0].set_title("Detections over time")
axes[0].grid(alpha=0.3)

# 2. all points from the whole capture overlaid - a rough "walked path" view
all_pts = np.vstack([p for p in points if p is not None and len(p)])
sc = axes[1].scatter(all_pts[:, 0], all_pts[:, 1], c=all_pts[:, 3],
                      cmap="coolwarm", s=8, alpha=0.6)
plt.colorbar(sc, ax=axes[1], label="velocity [m/s]")
axes[1].set_xlabel("x [m]")
axes[1].set_ylabel("y [m]")
axes[1].set_title(f"All {len(all_pts)} detected points, colored by velocity")
axes[1].grid(alpha=0.3)
axes[1].set_aspect("equal")

plt.tight_layout()
out = "capture_summary.png"
plt.savefig(out, dpi=150)
print(f"\nSaved figure -> {out}")
plt.show()