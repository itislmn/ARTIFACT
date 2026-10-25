"""
display.py - a window from the standard library (tkinter): show an image, collect keys (computer_vision's display).

    from backend.display import Window
"""
from __future__ import annotations

import tkinter as tk

import numpy as np

from backend.imgio import ppm_bytes


class Window:
    def __init__(self, title: str):
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root, bd=0)
        self.label.pack()
        self.alive = True
        self._keys, self._clicks, self._scale = [], [], 1.0
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Key>", lambda e: self._keys.append(e.keysym.lower() if len(e.keysym) > 1 else e.char))
        self.label.bind("<Button-1>", lambda e: self._clicks.append((e.x / self._scale, e.y / self._scale)))

    def show(self, rgb: np.ndarray, scale: float = 1.0):
        self._scale = scale
        self._photo = tk.PhotoImage(data=ppm_bytes(rgb), format="PPM")
        self.label.configure(image=self._photo)
        self.update()

    def update(self):
        if self.alive:
            self.root.update()

    def keys(self) -> list[str]:
        self.update()
        k, self._keys = self._keys, []
        return k

    def clicks(self) -> list[tuple[float, float]]:
        self.update()
        c, self._clicks = self._clicks, []
        return c

    def title(self, text: str):
        if self.alive:
            self.root.title(text)

    def close(self):
        if self.alive:
            self.alive = False
            self.root.destroy()
