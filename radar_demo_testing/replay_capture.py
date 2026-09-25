"""
Replay a saved capture as an animation - see exactly what happened during
recording, frame by frame, after the fact.

View it live on screen:
    python replay_capture.py uart_capture.npz

Save it as a video file you can play anywhere (no Python needed to show it -
this is the safest option for a meeting room with an unfamiliar projector):
    python replay_capture.py uart_capture.npz --save captures/replay.mp4

If you don't have ffmpeg installed, it automatically falls back to a .gif:
    python replay_capture.py uart_capture.npz --save replay.gif

Options:
    --speed 2       play back at 2x speed (default 1x)
    --trail 5       how many recent frames to keep visible at once (fading)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--save", default=None, help="output .mp4 or .gif instead of showing live")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--trail", type=int, default=5)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    frame_ids = data["frame_ids"]
    num_obj = data["num_obj"]
    points = data["points"]

    n = len(frame_ids)
    print(f"Loaded {n} frames from {args.npz}")

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title(f"Replay - {args.npz}")
    sc = ax.scatter([], [], s=30, c=[], cmap="plasma", vmin=-3, vmax=3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m] (range)")
    ax.set_title("Radar capture replay")
    ax.grid(alpha=0.3)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                   fontsize=11, family="monospace",
                   bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("velocity [m/s] (negative = approaching)")

    def update(i):
        lo = max(0, i - args.trail)
        recent = [points[j] for j in range(lo, i + 1) if points[j] is not None and len(points[j])]
        if recent:
            allp = np.vstack(recent)
            sc.set_offsets(np.c_[allp[:, 0], allp[:, 1]])
            sc.set_array(allp[:, 3])
        else:
            sc.set_offsets(np.empty((0, 2)))
        txt.set_text(f"frame {frame_ids[i]}\nobjects {num_obj[i]}\n"
                      f"{i+1}/{n}")
        return sc, txt

    interval_ms = max(1, int(1000 / (10 * args.speed)))  # assume ~10 Hz capture
    anim = FuncAnimation(fig, update, frames=n, interval=interval_ms, blit=False)

    if args.save:
        print(f"Rendering to {args.save} ... (this can take a minute)")
        if args.save.lower().endswith(".gif"):
            anim.save(args.save, writer=PillowWriter(fps=10 * args.speed))
        else:
            try:
                anim.save(args.save, writer="ffmpeg", fps=10 * args.speed)
            except Exception as e:
                fallback = args.save.rsplit(".", 1)[0] + ".gif"
                print(f"  ffmpeg not available ({e}); saving as {fallback} instead")
                anim.save(fallback, writer=PillowWriter(fps=10 * args.speed))
        print("Done. This file plays in any video/gif viewer, standalone.")
    else:
        plt.show()


if __name__ == "__main__":
    main()