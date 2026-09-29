r"""
LESSON 05 - What a raw sample actually IS, and where "IQ" really comes from

    python 05_adc_sample_format_and_iq.py

No hardware needed - this is byte-level arithmetic and one FFT.

------------------------------------------------------------------------
WHAT THE ADC IS ACTUALLY DIGITIZING
------------------------------------------------------------------------
An FMCW chirp radar never digitizes the raw 77 GHz signal directly - way
too fast for any practical ADC. Instead, the received echo is mixed
("beat") against a copy of the currently-transmitting chirp. Because the
echo is a slightly time-delayed copy of the transmitted ramp, and both
are sweeping frequency at the same rate, that mixing produces a much
slower "beat" tone whose frequency is directly proportional to the
target's range (further target -> more delay -> higher beat frequency).
THAT beat signal - megahertz, not gigahertz - is what the ADC digitizes,
once per sample, numAdcSamples times per chirp, for every enabled RX
channel. Each individual sample is just an integer proportional to that
beat signal's instantaneous voltage at that instant. Lesson 07 is where
that beat frequency actually gets turned into a range number via FFT;
this lesson is only about the bytes themselves.

------------------------------------------------------------------------
REAL vs COMPLEX SAMPLING - TWO DIFFERENT BYTE LAYOUTS
------------------------------------------------------------------------
adcCfg's adcOutputFmt (lesson 01) picks between:

  0 = REAL: one int16 per sample. That's it - just a plain sequence of
      16-bit signed integers, one per (chirp, RX channel, time-sample).
      THIS is the format this hardware (AWR2944P) actually uses and
      that every real capture in this project has produced -
      empirically validated many times over.

  1/2 = COMPLEX (1x/2x): the on-chip DDC (digital down-converter)
      produces an in-phase (I) and quadrature (Q) value per sample
      instead of one plain value, doubling the raw byte count per
      sample. This is the default most TI documentation assumes, and
      is standard on older chips (xWR16xx/18xx/68xx) - but is NOT a
      supported raw-DCA1000-capture format on the AWR2944 family in TDM
      mode (lesson 01/03). The byte layout shown for it below is
      included for conceptual completeness (and because you'll see it
      in TI's general documentation) but has NOT been exercised against
      real bytes from this hardware in this project, unlike the real-
      format path, which has - said plainly so you know which parts of
      this lesson to trust to what degree.

------------------------------------------------------------------------
THE IMPORTANT, COUNTER-INTUITIVE PART
------------------------------------------------------------------------
"Real sampling" sounds like it should mean "you lose the phase/IQ
information". It doesn't. Phase information about a signal's frequency
content doesn't come from having two ADC channels (I and Q) - it comes
from the FOURIER TRANSFORM. Feed a real-valued time series into an FFT
and you get back a COMPLEX-valued spectrum: each frequency bin has both
a magnitude and a phase, full stop, regardless of whether the input was
"real" or "complex" sampled. This is not a special property of radar
data - it's a basic, textbook fact about the discrete Fourier transform
of any real-valued signal (look up "Hermitian symmetry of the DFT" if
you want the formal statement). The demonstration at the bottom of this
file proves it with actual numbers, not just this paragraph's word for
it.

What real sampling DOES cost you: because a real signal's spectrum is
symmetric (bin k and bin N-k carry the same information, just complex
conjugates of each other), only HALF of your FFT's bins are actually
independent, new information. So for the same raw sample count N, a
real-sampled system gives you N/2 usable range bins where a complex-
sampled system would give you N. That's a real, concrete cost - just
not the one people usually assume ("losing IQ/phase") when they hear
"real, not complex".
"""
import struct

import os
import tempfile

import numpy as np


def explain_byte_layout():
    print("=" * 70)
    print("BYTE LAYOUT - REAL format (what this project's hardware sends)")
    print("=" * 70)
    print("""
    byte:     0    1     2    3     4    5     ...
              [int16 #0]  [int16 #1]  [int16 #2]  ...
              (one sample, no pairing)

    A .bin file is just this, back to back, for:
        numAdcSamples x numRx x chirpsPerFrame x numFrames

    total int16 values. No headers, no per-sample metadata - all the
    structure (which sample belongs to which chirp/RX/frame) is
    POSITIONAL, which is exactly why lesson 06's reshape has to get the
    axis order exactly right, or you silently scramble which sample
    belongs to which channel without any error being raised.
    """)

    print("=" * 70)
    print("BYTE LAYOUT - COMPLEX format (NOT used by this hardware; shown")
    print("for contrast only, per TI's general documentation - NOT")
    print("independently verified against real captured bytes in this")
    print("project)")
    print("=" * 70)
    print("""
    byte:     0    1     2    3     4    5     6    7
              [ I#0 ]   [ I#1 ]    [ Q#0 ]   [ Q#1 ]
              (two samples' I values, then their Q values, interleaved
               in groups of 2 - a consequence of how 4 LVDS lanes pack
               samples, not a simple "IQIQIQ" alternation)
    """)


def demonstrate_real_format_roundtrip():
    print("=" * 70)
    print("DEMONSTRATION 1 - a tiny fake 'real format' capture, read back")
    print("exactly the way capture_lib.py's load_complex_cube() does")
    print("=" * 70)
    # 8 fake samples, real format: just int16s.
    fake_samples = np.array([100, -50, 200, 0, -300, 75, 150, -25], dtype=np.int16)
    bin_path = os.path.join(tempfile.gettempdir(), "lesson05_fake_real.bin")
    fake_samples.tofile(bin_path)

    raw = np.fromfile(bin_path, dtype=np.int16)
    # This is the REAL branch of load_complex_cube(): just cast straight
    # into the real part of a complex array. No pairing, no reshuffling.
    data = raw.astype(np.complex64)
    print(f"Raw int16 values read back : {raw}")
    print(f"As complex64 (real branch) : {data}")
    print(f"Imaginary part             : {data.imag} <- exactly zero, as expected")
    print("(this is BEFORE any FFT - the complex dtype here is just a")
    print("convenient container so the SAME downstream code, from")
    print("lesson 06 onward, works whether the raw format was real or")
    print("complex; it says nothing about phase yet)")


def demonstrate_fft_produces_phase_from_real_input():
    print("\n" + "=" * 70)
    print("DEMONSTRATION 2 - a REAL-valued signal's FFT is still complex,")
    print("with genuine, non-trivial phase - proving the claim above with")
    print("numbers instead of just asserting it")
    print("=" * 70)
    N = 64
    fs = 1000.0          # Hz, arbitrary
    f_beat = 100.0        # Hz - stand-in for "range" (lesson 07 makes this real)
    true_phase = 1.234    # radians - stand-in for whatever phase a real
    #                        target/channel would impose (lesson 08 turns
    #                        exactly this kind of phase into an angle)
    t = np.arange(N) / fs
    real_signal = np.cos(2 * np.pi * f_beat * t + true_phase)  # PURELY REAL
    assert np.all(np.isreal(real_signal))

    spectrum = np.fft.fft(real_signal)
    bin_idx = int(round(f_beat * N / fs))
    measured_phase = np.angle(spectrum[bin_idx])
    mirror_bin = N - bin_idx
    measured_phase_mirror = np.angle(spectrum[mirror_bin])

    print(f"Input: a plain, 100%-real cosine wave. dtype={real_signal.dtype}")
    print(f"FFT bin {bin_idx} (the real target frequency):")
    print(f"    magnitude = {abs(spectrum[bin_idx]):.2f}")
    print(f"    phase     = {measured_phase:+.4f} rad")
    print(f"Mirror bin {mirror_bin} (= N - {bin_idx}, the Hermitian partner):")
    print(f"    magnitude = {abs(spectrum[mirror_bin]):.2f}  <- SAME magnitude")
    print(f"    phase     = {measured_phase_mirror:+.4f} rad <- NEGATED phase")
    print(f"\nThe FFT recovered a phase - from a signal that was never")
    print(f"'complex' at any point before the transform. That's the")
    print(f"Hermitian symmetry lesson 01 referenced: bin {bin_idx} and bin")
    print(f"{mirror_bin} carry the same information (conjugates of each")
    print(f"other), which is exactly why real sampling only gives you")
    print(f"N/2 = {N//2} INDEPENDENT bins out of N = {N} total, even though")
    print(f"every individual bin you DO use is a full, genuine complex")
    print(f"number with real phase in it.")


if __name__ == "__main__":
    explain_byte_layout()
    demonstrate_real_format_roundtrip()
    demonstrate_fft_produces_phase_from_real_input()
    print("\nNext: 06_radar_cube_and_tdm_mimo.py - turning a long flat")
    print("stream of these samples into a (frame, chirp, channel, sample)")
    print("cube, and TDM-MIMO: how firing 4 TX antennas one at a time")
    print("fakes a much bigger antenna array.")
