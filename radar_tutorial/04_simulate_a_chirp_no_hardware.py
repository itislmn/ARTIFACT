"""
LESSON 04 — Build intuition with FAKE data you can fully trust

No hardware, no .cfg file. Before trusting real radar data, let's build a
completely synthetic version of the exact same math, where WE know the
right answer in advance, so we can verify the FFT actually does what
lesson 02 claimed.

The story: imagine a single, perfectly reflective object sitting exactly
3.0 meters away. We MANUALLY compute what the ADC would see (a slowly
oscillating wave at a specific "beat frequency"), then FFT it, and check
that the peak really does land at 3.0 m.

Run it:
    python 04_simulate_a_chirp_no_hardware.py
(no arguments needed - everything here is made up on purpose)
"""
import os

import matplotlib.pyplot as plt
import numpy as np

C = 299792458.0
OUT_DIR = "output/04_simulate_a_chirp_no_hardware"

# --- pretend radar settings, similar magnitude to a real profileCfg ---
start_freq_hz = 77e9
slope_hz_per_s = 60e12          # 60 MHz/us, a typical chirp slope
num_samples = 512
sample_rate_hz = 10e6           # 10 Msps
true_range_m = 3.0              # <-- the "ground truth" we're pretending exists

print("Simulating one chirp reflecting off an object at exactly "
      f"{true_range_m} m...\n")

# --- the physics: round-trip delay causes a constant beat frequency ---
round_trip_s = 2 * true_range_m / C
beat_freq_hz = slope_hz_per_s * round_trip_s
print(f"Round trip time    : {round_trip_s*1e9:.2f} ns")
print(f"Predicted beat freq: {beat_freq_hz/1e3:.2f} kHz")
print("(This is the ONE frequency we expect the FFT to find a peak at.)")

# --- build the fake ADC samples: just a sine wave at that beat frequency ---
t = np.arange(num_samples) / sample_rate_hz
noise = 0.05 * np.random.randn(num_samples)   # tiny noise, so it looks like real data
adc_samples = np.sin(2 * np.pi * beat_freq_hz * t) + noise

# --- the actual lesson: one FFT, and read off where the peak is ---
window = np.hanning(num_samples)
spectrum = np.abs(np.fft.fft(adc_samples * window))
freqs = np.fft.fftfreq(num_samples, d=1/sample_rate_hz)
half = num_samples // 2

# convert each frequency bin to the range it corresponds to
range_axis_m = freqs[:half] * C / (2 * slope_hz_per_s)

peak_bin = np.argmax(spectrum[:half])
measured_range_m = range_axis_m[peak_bin]

print(f"\nMeasured range from the FFT peak: {measured_range_m:.3f} m")
print(f"True range we simulated         : {true_range_m:.3f} m")
print("If these two numbers are close, you've just proven to yourself that")
print("'take an FFT of the beat signal, the peak location is the range'")
print("(lesson 01's core claim) is actually true, not just something I told you.")

os.makedirs(OUT_DIR, exist_ok=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(t * 1e6, adc_samples)
a1.set_xlabel("time [us]"); a1.set_ylabel("ADC value")
a1.set_title("Fake raw ADC samples (one chirp)")
a1.grid(alpha=0.3)

a2.plot(range_axis_m, spectrum[:half])
a2.axvline(true_range_m, color="red", linestyle="--", label=f"true range = {true_range_m}m")
a2.set_xlabel("range [m]"); a2.set_ylabel("FFT magnitude")
a2.set_title("Range FFT - peak should sit on the red line")
a2.legend(); a2.grid(alpha=0.3)

plt.tight_layout()
out = f"{OUT_DIR}/simulated_range_fft.png"
plt.savefig(out, dpi=150)
print(f"\nSaved {out}")
print("\nTRY THIS: change true_range_m at the top of this file to 1.5 or 8.0,")
print("re-run, and confirm the peak follows. That's the whole idea internalized.")
