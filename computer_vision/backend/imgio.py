"""
imgio.py - images and videos without image libraries: PNG with zlib, MP4 through an ffmpeg process fed by a writer
thread.

    from backend.imgio import read_png, write_png, read_video, VideoWriter
"""
from __future__ import annotations

import queue
import shutil
import struct
import subprocess
import threading
import zlib
from pathlib import Path

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path, img: np.ndarray, dpi: float | None = None, level: int = 1):
    img = np.ascontiguousarray(img, np.uint8)
    h, w = img.shape[:2]
    colour = 0 if img.ndim == 2 else 2
    rows = np.zeros((h, 1 + img[0].size), np.uint8)
    rows[:, 1:] = img.reshape(h, -1)
    png = PNG_SIGNATURE + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, colour, 0, 0, 0))
    if dpi:
        ppm = int(round(dpi / 0.0254))
        png += _chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))
    png += _chunk(b"IDAT", zlib.compress(rows.tobytes(), level)) + _chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def read_png(path) -> np.ndarray:
    data = Path(path).read_bytes()
    if data[:8] != PNG_SIGNATURE:
        raise ValueError(f"{path}: not a PNG")
    pos, idat = 8, []
    while pos < len(data):
        n, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h, depth, colour = struct.unpack(">IIBB", body[:10])
        elif tag == b"IDAT":
            idat.append(body)
        pos += 12 + n
    if depth != 8 or colour not in (0, 2):
        raise ValueError(f"{path}: only 8-bit grey/RGB PNGs written by imgio are supported")
    ch = 1 if colour == 0 else 3
    rows = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(h, 1 + w * ch)
    if rows[:, 0].any():
        raise ValueError(f"{path}: PNG row filters in use; only PNGs written by imgio are supported")
    img = rows[:, 1:].reshape(h, w, ch)
    return img[:, :, 0] if ch == 1 else img


def ppm_bytes(rgb: np.ndarray) -> bytes:
    h, w = rgb.shape[:2]
    return f"P6 {w} {h} 255\n".encode() + np.ascontiguousarray(rgb, np.uint8).tobytes()


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg") or (r"C:\Program Files\ffmpeg\ffmpeg.exe"
                                      if Path(r"C:\Program Files\ffmpeg\ffmpeg.exe").exists() else None)
    if exe is None:
        raise RuntimeError("ffmpeg not found (needed for MP4 only; PNG snapshots work without it)")
    return exe


class VideoWriter:

    def __init__(self, path, size_wh, fps: float, crf: int = 16):
        w, h = size_wh
        cmd = [ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "ultrafast",
               "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)]
        self.path, self.n = Path(path), 0
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._queue = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._run, name="ffmpeg-writer", daemon=True)
        self._thread.start()

    def _run(self):
        while (item := self._queue.get()) is not None:
            rgb, repeat = item
            data = np.ascontiguousarray(rgb, np.uint8).tobytes()
            for _ in range(repeat):
                self._proc.stdin.write(data)
        self._proc.stdin.close()

    def write(self, rgb: np.ndarray, repeat: int = 1):
        self._queue.put((rgb, repeat))
        self.n += repeat

    def close(self):
        self._queue.put(None)
        self._thread.join()
        self._proc.wait()


def read_video(path):
    exe = ffmpeg_exe()
    probe = subprocess.run([str(Path(exe).with_name("ffprobe")), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, check=True)
    w, h = map(int, probe.stdout.strip().split(","))
    proc = subprocess.Popen([exe, "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * 3
    try:
        while len(buf := proc.stdout.read(n)) == n:
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()
