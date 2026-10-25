"""
text_reading.py - the text on a radar-tracked object, read at the radar's position and scale with computer_vision's
readers (shared with obstacle/).

    from methods.text_reading import read_object
"""
from __future__ import annotations

import cv2
import numpy as np

TILTS_DEG = (0, -10, 10, -20, 20)


def half_size(range_m: float, f_px: float, rcfg: dict, r_obj: float) -> int:
    s = f_px / range_m
    return int(0.6 * np.hypot(*rcfg["text"]["region_m"]) * s + r_obj * s) + 8


def patch_around(rgb, centre_uv, range_m, f_px, rcfg, r_obj):
    half = half_size(range_m, f_px, rcfg, r_obj)
    cu, cv_ = int(round(centre_uv[0])), int(round(centre_uv[1]))
    pad = cv2.copyMakeBorder(rgb, half, half, half, half, cv2.BORDER_REPLICATE)
    patch = np.ascontiguousarray(pad[cv_:cv_ + 2 * half, cu:cu + 2 * half])
    return patch, (half + centre_uv[0] - cu, half + centre_uv[1] - cv_)


def read_object(reader, rgb, centre_uv, range_m, f_px, rcfg, r_obj):
    patch, c = patch_around(rgb, centre_uv, range_m, f_px, rcfg, r_obj)
    return read_patch(reader, patch, c, range_m, f_px, rcfg, r_obj)


def read_patch(reader, patch, c, range_m, f_px, rcfg, r_obj):
    s = f_px / range_m
    best = None
    for ang in TILTS_DEG:
        img = patch if ang == 0 else cv2.warpAffine(patch, cv2.getRotationMatrix2D(c, ang, 1.0), patch.shape[1::-1],
                                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        for dz in (0.0, r_obj / 2, -r_obj / 2):
            rd = reader.read(img, (c[0], c[1] - dz * s), range_m, f_px, rcfg)
            conf = float(np.mean(rd.margins)) if len(rd.margins) else -np.inf
            if best is None or conf > best[0]:
                best = (conf, rd)
    return best[1]
