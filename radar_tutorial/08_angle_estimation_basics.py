r"""
LESSON 08 - Turning phase differences across the virtual array into a bearing

    python 08_angle_estimation_basics.py

No hardware needed - synthesizes a target at a KNOWN angle and recovers
it, asserting the recovered angle is close to the truth.

------------------------------------------------------------------------
A NOTE ON THE ARRAY GEOMETRY USED HERE
------------------------------------------------------------------------
Lesson 06 already covered, honestly, that this project does not have a
fully-confirmed field-by-field decode of the real antGeometryCfg line
beyond its trailing spacing values (0.5 lambda, 0.8 lambda - confirmed
against the EVM's own user guide) and the observation that only 2 of
the 4 physical TX appear to feed the main azimuth array.

Rather than build this lesson's math on top of that uncertainty, the
geometry below is a CLEAN, STANDARD, textbook virtual-ULA construction -
2 TX at (0, 2.0 lambda) combined with 4 RX at (0, 0.5, 1.0, 1.5 lambda) -
which is a very common, well-documented way to get a gap-free, evenly-
spaced (0.5 lambda) 8-element virtual array from 2 TX x 4 RX, and is
consistent with (though not a certified re-derivation of) what lesson 06
observed. The ANGLE MATH below is correct and general; only the exact
element positions would need to change to match a fully-confirmed real
geometry for this specific board.

------------------------------------------------------------------------
THE PHYSICS: WHY PHASE ENCODES ANGLE
------------------------------------------------------------------------
A target far enough away sends back a signal that looks like a PLANE
WAVE by the time it reaches the array (all antenna elements see the
same wavefront, just shifted in time/phase depending on their exact
position). For an array of elements laid out along one line, spaced d
apart, a plane wave arriving from angle theta (measured from broadside,
i.e. straight ahead = 0 deg) reaches each successive element with an
extra path length of d*sin(theta), which is an extra phase of
2*pi*d*sin(theta)/lambda. Measure that phase difference across the
array and you can solve for theta - that's the entire idea behind every
angle-of-arrival technique, from the simple scan below to MUSIC/ESPRIT/
Capon (more advanced methods for the same underlying physics).
"""
import numpy as np

C = 2.99792458e8


def build_virtual_ula():
    """8 virtual elements: (tx_id, rx_id, position_in_wavelengths). See
    the module docstring for why these specific numbers."""
    tx_pos = [0.0, 2.0]
    rx_pos = [0.0, 0.5, 1.0, 1.5]
    elements = []
    for tx_id, tx_x in enumerate(tx_pos):
        for rx_id, rx_x in enumerate(rx_pos):
            elements.append((tx_id, rx_id, tx_x + rx_x))
    elements.sort(key=lambda e: e[2])
    positions = [e[2] for e in elements]
    spacings = np.diff(positions)
    assert np.allclose(spacings, 0.5), \
        f"expected a uniform 0.5-lambda virtual ULA, got spacings {spacings}"
    print("Virtual ULA (sorted by position, in wavelengths):")
    for tx_id, rx_id, pos in elements:
        print(f"  TX{tx_id}-RX{rx_id}  ->  {pos:.1f} lambda")
    return elements


def bartlett_beamform(snapshot, positions_lambda, angles_deg):
    """The simplest angle-of-arrival estimator there is: for each
    candidate angle, build the phase pattern a PLANE WAVE from that
    angle would have produced across the array (the "steering vector"),
    phase-align the measured snapshot against it, and sum. A wrong
    candidate angle sums mostly-cancelling phases (destructive
    interference, small result); the RIGHT candidate angle sums
    in-phase (constructive interference, a big peak). Scan enough
    candidate angles and take the peak.

    snapshot: complex array, one value per virtual element (e.g. the
        range/Doppler-FFT'd value at the target's bin, one per channel -
        lesson 07 shows how to get this for a single channel; here we
        need it for every channel).
    positions_lambda: element positions, in wavelengths, same order as
        snapshot.
    """
    # Sign convention: a plane wave truly arriving from angle theta
    # induces phase exp(+j*2*pi*pos*sin(theta)) at each element (that's
    # what synthesize_snapshot() below builds). To coherently sum a
    # snapshot that actually came from theta, multiply by the CONJUGATE
    # of that, exp(-j*2*pi*pos*sin(theta)), so the phase cancels to zero
    # (constructive) exactly when the candidate theta matches the truth.
    positions = np.array(positions_lambda)
    spectrum = np.zeros(len(angles_deg))
    for i, ang in enumerate(angles_deg):
        theta = np.radians(ang)
        compensating_phase = np.exp(-1j * 2 * np.pi * positions * np.sin(theta))
        spectrum[i] = np.abs(np.sum(compensating_phase * snapshot)) ** 2
    return spectrum


def synthesize_snapshot(elements, true_angle_deg, snr_db=20.0, seed=0):
    """A simplified, angle-only synthetic snapshot (skips range/Doppler -
    lesson 07 already proved those independently; this isolates the
    angle math). Each element just gets the plane-wave phase for the
    true angle, plus noise."""
    rng = np.random.default_rng(seed)
    positions = np.array([e[2] for e in elements])
    theta = np.radians(true_angle_deg)
    clean = np.exp(1j * 2 * np.pi * positions * np.sin(theta))
    noise_power = 1.0 / (10 ** (snr_db / 10))
    noise = (rng.normal(0, np.sqrt(noise_power / 2), clean.shape)
             + 1j * rng.normal(0, np.sqrt(noise_power / 2), clean.shape))
    return clean + noise


def main():
    elements = build_virtual_ula()
    positions = [e[2] for e in elements]

    true_angle = 22.0  # degrees off boresight
    snapshot = synthesize_snapshot(elements, true_angle, snr_db=20.0)

    angles_deg = np.arange(-90, 90.05, 0.1)
    spectrum = bartlett_beamform(snapshot, positions, angles_deg)
    est_angle = angles_deg[int(np.argmax(spectrum))]

    print(f"\nTrue angle:      {true_angle:+.1f} deg")
    print(f"Estimated angle: {est_angle:+.1f} deg (0.1 deg scan resolution)")
    print(f"Error:           {abs(est_angle - true_angle):.2f} deg")
    assert abs(est_angle - true_angle) < 1.0, \
        "Bartlett beamformer should localize a single clean target to well under 1 degree"

    print("\n(For reference: this 8-element, 0.5-lambda array's null-to-")
    print("null beamwidth near broadside is roughly 2/N radians "
          f"({np.degrees(2/len(elements)):.0f} deg) - that's the "
          "RESOLUTION for telling two close targets apart, which is much")
    print("coarser than the LOCALIZATION precision for a single isolated")
    print("target shown above; those are two different numbers people")
    print("often conflate.")

    print("\nWhat a real 16-channel measurement adds beyond this 8-element")
    print("illustrative array: more elements (better resolution), a 2D")
    print("layout (azimuth AND elevation, not just one line), and -")
    print("critically - the REAL, exact element positions from a properly")
    print("confirmed antGeometryCfg decode (lesson 06) rather than this")
    print("lesson's clean textbook stand-in. The MATH above - build a")
    print("steering vector, correlate, scan, take the peak - is exactly")
    print("what changes only its `positions_lambda` input to go from this")
    print("demo to a real measurement.")


if __name__ == "__main__":
    main()
    print("\nNext: 09_realtime_active_sensing_loop.py - structuring all of")
    print("this (lessons 03-08) as a continuous acquire -> process -> ")
    print("decide -> act loop, which is what 'active sensing' actually ")
    print("means in code.")
