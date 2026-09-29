r"""
LESSON 06 - From a flat byte stream to a radar cube, and TDM-MIMO

    python 06_radar_cube_and_tdm_mimo.py

No hardware needed - this is reshaping arithmetic, proven with tagged
synthetic values so you can SEE exactly which byte ends up where.

------------------------------------------------------------------------
THE HIERARCHY (from lesson 01's parsed .cfg)
------------------------------------------------------------------------
One capture .bin file is, in order, nothing but:

    numFrames  x  chirpsPerFrame  x  numRx  x  numAdcSamples

int16 values (real format), where chirpsPerFrame = chirps_per_loop x
numLoops, and chirps_per_loop = chirpEndIdx - chirpStartIdx + 1 (how
many chirpCfg entries fire per loop - 4 in this project's cfg).

The nesting, slowest-changing to fastest-changing:
    frame
      loop (repeats the WHOLE chirp sequence this many times - your
            slow-time / Doppler axis)
        chirp-in-loop / "firing slot" (0..3 - which chirpCfg entry, in
            firing order, NOT necessarily physical TX id - see below)
          RX channel (0..3, physical receive antenna)
            ADC sample (0..numAdcSamples-1, fast-time / range axis)

reshape(n_frames, chirpsPerFrame, numRx, numAdcSamples), then splitting
chirpsPerFrame into (numLoops, chirps_per_loop), is a direct, literal
encoding of that nesting - IF the axis order assumption is right. Get
it wrong (e.g. assume RX varies before samples do, or chirps before
loops) and numpy will not raise an error. It'll happily give you back
an array of the right SHAPE, filled with completely scrambled data -
this is the single easiest way to silently corrupt a whole capture, and
exactly the kind of bug that would produce noisy-looking, non-physical
results without any exception ever being raised.

------------------------------------------------------------------------
TDM-MIMO: WHY CYCLING 4 TX ANTENNAS FAKES A BIGGER ARRAY
------------------------------------------------------------------------
Only ONE TX antenna transmits at any instant (Time-Division Multiplexed
- that's the "TDM" in TDM-MIMO). Within one loop, the 4 chirpCfg entries
fire on 4 different physical TX antennas, one chirp each, in sequence.

Here's the trick that makes this worth doing: a signal received on RX_j
after being transmitted from TX_i has traveled a path equivalent to a
SINGLE antenna sitting at the midpoint of TX_i and RX_j (to first
order, for a target far enough away). So instead of just 4 physical RX
elements, cycling through 4 TX gives you 4 x 4 = 16 DISTINCT virtual
antenna positions, from only 4+4=8 physical antennas - a denser virtual
array than you could physically fit RX elements for, which directly
improves angle resolution (lesson 08). This is the entire reason TDM-
MIMO is used at all; it is standard, well-established radar array
theory, not something specific to this project.

The cost: because the 4 TX chirps within a loop are NOT simultaneous,
each virtual channel's data was actually sampled at a very slightly
different instant (one chirp period apart, lesson 01's numbers put that
at ~243us for this cfg). For a genuinely fast-moving target this causes
a small phase error across TX in the angle estimate; for anything
roughly stationary during one loop (this project's antenna-pattern-
measurement use case, e.g.) it's negligible.

------------------------------------------------------------------------
THE REAL BUG THIS PROJECT HAD: FIRING SLOT != TX ID
------------------------------------------------------------------------
It is tempting to assume chirp-firing-slot 0,1,2,3 corresponds to
TX0,TX1,TX2,TX3 in that order. Lesson 01 already showed this project's
actual .cfg does NOT do that:

    chirpCfg 0 ... txEnable=1   -> slot 0 fires TX0
    chirpCfg 1 ... txEnable=4   -> slot 1 fires TX2
    chirpCfg 2 ... txEnable=8   -> slot 2 fires TX3
    chirpCfg 3 ... txEnable=2   -> slot 3 fires TX1

An earlier version of this project's code labeled virtual channels by
firing-slot index directly, so every plot's "TX2" column was actually
physical TX3's data, "TX3" was actually TX1's data, and so on. The
fix - proven below with tagged synthetic data so you can see it work -
is to build the permutation from chirpCfg's actual txEnable values and
apply it BEFORE collapsing the (slot, rx) axes into one "virtual
channel" axis.
"""
import numpy as np


# =======================================================================
# STEP 1 - recover the firing-slot -> physical-TX-id permutation from
# chirpCfg (same logic as lesson 01, isolated here since lesson 06 is
# where it actually gets USED).
# =======================================================================
def compute_tx_reorder(chirp_tx: dict, num_tx: int):
    """chirp_tx: {firing_slot_index: txEnable_bitmask}, straight from
    parsing chirpCfg lines. Returns perm such that perm[physical_tx_id]
    = firing_slot_index - i.e. "TX i's data lives in firing slot
    perm[i]", which is exactly what you index the cube's slot axis with
    to put it back in physical-TX order."""
    perm = [None] * num_tx
    for slot, mask in chirp_tx.items():
        if not mask:
            continue
        tx_id = mask.bit_length() - 1
        if tx_id < num_tx:
            perm[tx_id] = slot
    if None in perm:
        raise ValueError(f"could not resolve every TX id from {chirp_tx}")
    return perm


# =======================================================================
# STEP 2 - the reshape itself, with the reorder applied.
# =======================================================================
def bytes_to_cube(raw_int16: np.ndarray, n_frames, num_loops, chirps_per_loop,
                   num_rx, num_adc_samples, tx_reorder, num_tx):
    cube = raw_int16.astype(np.complex64).reshape(
        n_frames, num_loops * chirps_per_loop, num_rx, num_adc_samples)
    cube = cube.reshape(n_frames, num_loops, num_tx, num_rx, num_adc_samples)
    cube = cube[:, :, tx_reorder, :, :]     # <-- the fix
    cube = cube.reshape(n_frames, num_loops, num_tx * num_rx, num_adc_samples)
    return cube   # (frame, loop, virtual_channel[=tx*num_rx+rx], sample)


# =======================================================================
# PROOF - tagged synthetic data. Every (slot, rx) block is filled with a
# distinct, recognizable number: slot*100 + rx. If the reorder is right,
# reading out virtual channel ch should show tx=ch//num_rx, rx=ch%num_rx,
# and the VALUE at that channel should equal (firing_slot_for_that_tx)*100
# + rx - proving the physical TX's data, not just the firing-slot's data,
# ended up at the position labeled with that TX's id.
# =======================================================================
def prove_it():
    chirp_tx = {0: 1, 1: 4, 2: 8, 3: 2}   # this project's real firing order
    num_tx, num_rx, num_adc = 4, 4, 2
    num_loops = 1

    tagged = np.zeros((1, num_loops, num_tx, num_rx, num_adc), dtype=np.int16)
    for slot in range(num_tx):
        for rx in range(num_rx):
            tagged[0, 0, slot, rx, :] = slot * 100 + rx
    raw = tagged.reshape(-1)

    perm = compute_tx_reorder(chirp_tx, num_tx)
    print(f"chirpCfg firing order (slot -> txEnable mask): {chirp_tx}")
    print(f"Recovered permutation (perm[tx_id] = firing slot): {perm}\n")

    print(f"{'channel':10s} {'tx':4s} {'rx':4s} {'value':>7s}  {'means':30s}")
    print("-" * 60)
    cube_wrong = bytes_to_cube(raw, 1, num_loops, num_tx, num_rx, num_adc,
                                tx_reorder=list(range(num_tx)), num_tx=num_tx)
    cube_right = bytes_to_cube(raw, 1, num_loops, num_tx, num_rx, num_adc,
                                tx_reorder=perm, num_tx=num_tx)
    any_wrong = False
    for ch in range(num_tx * num_rx):
        tx, rx = ch // num_rx, ch % num_rx
        v_wrong = int(cube_wrong[0, 0, ch, 0].real)
        v_right = int(cube_right[0, 0, ch, 0].real)
        expected = perm[tx] * 100 + rx   # the value physical TX `tx` actually wrote
        ok = v_right == expected
        any_wrong = any_wrong or (v_wrong != expected)
        flag = "" if ok else "  <-- WRONG"
        note = (f"labeled TX{tx}-RX{rx}, fixed reads slot {perm[tx]}'s "
                f"data (correct){flag}")
        print(f"ch={ch:2d}      TX{tx}   RX{rx}   {v_right:7d}  {note}")

    print(f"\nWithout the reorder (naive firing-slot-as-tx-id), the SAME")
    print(f"channel labeled 'TX2' would instead show value "
          f"{int(cube_wrong[0,0,2*num_rx,0].real)} - physical TX3's data "
          f"(firing slot 2), not TX2's. {'This mismatch is the bug.' if any_wrong else ''}")
    assert all(int(cube_right[0, 0, ch, 0].real) == perm[ch // num_rx] * 100 + ch % num_rx
               for ch in range(num_tx * num_rx)), "reorder should recover exact tags"
    print("\nAssertion passed: every physical TX's tagged data lands at the")
    print("channel index labeled with ITS OWN id, not its firing order.")


# =======================================================================
# antGeometryCfg - what it's FOR, and an honest note on what's certain
# vs uncertain about its exact fields for this specific board.
# =======================================================================
def discuss_ant_geometry_cfg():
    line = ("antGeometryCfg 1 0 1 1 1 2 1 3 0 2 0 3 0 4 0 5 1 4 1 5 1 6 1 7 "
            "1 8 1 9 1 10 1 11 0.5 0.8")
    print("\n" + "=" * 70)
    print("antGeometryCfg - the physical array layout")
    print("=" * 70)
    print(f"Real line from this project's .cfg:\n  {line}\n")
    print("What this command is FOR: telling the chip's own on-chip angle")
    print("estimator (and, when we parse it ourselves, OUR angle estimator")
    print("in lesson 08) the actual physical positions of each virtual")
    print("TX/RX antenna combination, in units of wavelength - because")
    print("angle-of-arrival math (lesson 08) needs the REAL spacing, not")
    print("an assumed textbook one.")
    print()
    print("CONFIRMED (cross-checked against two independent sources):")
    print("  The last two values, 0.5 and 0.8, match the AWR2944EVM's own")
    print("  user guide, which states the antenna module's spacing as")
    print("  'lambda/2' and '0.8 lambda/2' - i.e. these are almost")
    print("  certainly (horizontal spacing, vertical spacing) in units of")
    print("  wavelength. Source: TI EVM User's Guide (spruj22c), section")
    print("  on the antenna module.")
    print()
    print("OBSERVED PATTERN (structurally suggestive, not authoritative):")
    print("  The 32 middle values group into 16 (a, b) pairs - matching")
    print("  4 TX x 4 RX = 16 virtual channels. Only a in {0, 1} ever")
    print("  appears (never 2 or 3), while b ranges from 1 to 11. That's")
    print("  consistent with only 2 of the 4 physical TX antennas feeding")
    print("  the main azimuth-line array (the other 2 likely dedicated to")
    print("  elevation sensing via a vertical offset - a common MIMO")
    print("  array design pattern), and b indexing roughly 11-12 distinct")
    print("  virtual azimuth positions - which lines up with the ~12-")
    print("  element azimuth ULA this project's earlier DOA work derived")
    print("  from this exact line.")
    print()
    print("WHAT I WON'T DO: assert a confident field-by-field spec for the")
    print("full command beyond what's shown above. I don't have TI's exact")
    print("antGeometryCfg documentation for this SDK/device pinned down,")
    print("and after the adcCfg mix-up earlier in this project, guessing")
    print("confidently here would repeat that mistake. If you need the")
    print("authoritative field layout: the mmWave SDK's own source (look")
    print("for antenna_geometry.c or the CLI command table in the demo's")
    print("source, if you have SDK access) is the real answer, or your")
    print("earlier DOA-suite project's own geometry-parsing code, which")
    print("was already validated against this array for real angle")
    print("estimates. Lesson 08 uses a clean, standard textbook virtual-")
    print("ULA layout for teaching the ANGLE MATH itself, explicitly")
    print("flagged as illustrative rather than a re-decode of this line -")
    print("the math is identical either way, only the exact element")
    print("positions you'd plug in for a real measurement would differ.")


if __name__ == "__main__":
    prove_it()
    discuss_ant_geometry_cfg()
    print("\nNext: 07_range_doppler_cfar.py - turning this correctly-")
    print("labeled cube into an actual range/velocity picture.")
