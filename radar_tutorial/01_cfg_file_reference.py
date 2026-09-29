r"""
LESSON 01 - The .cfg file, parameter by parameter

    python 01_cfg_file_reference.py                         (uses the bundled example cfg)
    python 01_cfg_file_reference.py --cfg mycustom.cfg

No hardware needed - this just reads a text file and does arithmetic.

------------------------------------------------------------------------
WHAT THIS FILE ACTUALLY IS
------------------------------------------------------------------------
The .cfg file is a script of plain-ASCII CLI commands, sent one line at a
time over UART to the radar's onboard MCU (lesson 02 covers HOW they're
sent). Nothing here is TI-magic: it's the exact same kind of thing as
typing commands into any serial console. The MCU's firmware ("mmWave
demo") parses each line, validates it, configures its hardware
accordingly, and replies "Done" or an error string.

Order matters. sensorStop/flushCfg must come first (undo whatever the
last session left running); the last line, sensorStart, is the one that
actually arms everything and starts chirping - every line before it is
just accumulating configuration that only takes effect once sensorStart
fires.

------------------------------------------------------------------------
A NOTE ON CONFIDENCE
------------------------------------------------------------------------
Every field explained below is either (a) confirmed against TI's own
mmWave SDK documentation, or (b) confirmed empirically by this project's
own working captures. Where I genuinely could not pin a field down with
confidence (this device's frameCfg has one MORE numeric field than TI's
published reference lists), I say so explicitly rather than guess - and
show you how to ask the firmware itself, which is the actually-correct
way to resolve device-specific ambiguity like that. Getting confidently
wrong information from an AI about this exact cfg file is literally what
caused a real bug earlier in this project (adcCfg 2 1 vs 2 0) - so this
lesson would rather under-claim than repeat that mistake.
"""
import argparse
import re

import numpy as np

C = 2.99792458e8  # speed of light, m/s


# =======================================================================
# THE PARSER
# =======================================================================
def parse_cfg_verbose(path, explain=True):
    """Walks the .cfg file. For every line it recognizes, prints a
    labeled breakdown of each field. Returns a dict of everything a DSP
    pipeline needs downstream (same keys later lessons use)."""
    p = {"chirpTx": {}, "numFrames": 0}
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith(("%", "#")):
            continue
        parts = re.split(r"\s+", line)
        cmd, v = parts[0], parts[1:]

        if cmd == "sensorStop":
            _say(explain, line, "Stop any running frame sequence and clear "
                 "the chip's state before re-configuring it. Always first.")

        elif cmd == "flushCfg":
            _say(explain, line, "Discard any config the chip is currently "
                 "holding in memory. Belt-and-braces alongside sensorStop.")

        elif cmd == "dfeDataOutputMode":
            mode = {1: "chirp-interleaved (single profile)",
                    2: "advanced/subframe (multiple profiles per frame)",
                    3: "continuous streaming"}.get(int(v[0]), "unknown")
            _say(explain, line, f"v[0]={v[0]} -> {mode}. This cfg uses mode "
                 "1, the simple case this whole tutorial assumes: one "
                 "profile, no subframes.")

        elif cmd == "channelCfg":
            rx_mask, tx_mask, cascade = int(v[0]), int(v[1]), int(v[2])
            p["numRx"] = bin(rx_mask).count("1")
            rx_bits = [i for i in range(4) if rx_mask & (1 << i)]
            tx_bits = [i for i in range(4) if tx_mask & (1 << i)]
            _say(explain, line,
                 f"rxChannelEn=0b{rx_mask:04b} -> RX antennas enabled: "
                 f"{[f'RX{i}' for i in rx_bits]} ({p['numRx']} of them).\n"
                 f"    txChannelEn=0b{tx_mask:04b} -> TX antennas made "
                 f"AVAILABLE (not yet 'firing' - that's chirpCfg's job "
                 f"below): {[f'TX{i}' for i in tx_bits]}.\n"
                 f"    cascading={cascade} -> 0 means single-chip mode "
                 "(this board, not a multi-chip cascade array).")

        elif cmd == "adcCfg":
            fmt = int(v[1])
            meaning = {0: "REAL (only the real part of the beat signal is "
                          "digitized and sent)",
                       1: "COMPLEX 1x (I/Q, image band filtered out)",
                       2: "COMPLEX 2x (I/Q, image band NOT filtered - "
                          "twice the samples for the same time)"}[fmt]
            note = ""
            if fmt == 0:
                note = ("\n    ^ This is the CORRECT setting for raw "
                        "DCA1000 capture on AWR2944-family chips in TDM "
                        "mode - confirmed on TI's E2E forum by a TI "
                        "engineer. Complex 1x/2x is the standard choice "
                        "on OLDER chips (xWR16xx/18xx/68xx) and is what "
                        "most TI documentation defaults to, but it is "
                        "NOT a supported raw-capture format on this chip "
                        "family - you'd either get an outright error or "
                        "silently zero LVDS output. See lesson 05 for "
                        "why using REAL here costs you nothing: the "
                        "range FFT of real data is still fully complex.")
            else:
                note = ("\n    ^ WARNING: on THIS chip family (AWR2944P), "
                        "raw DCA1000 capture needs adcCfg 2 0 (real), not "
                        "this. Expect zero LVDS output with this setting.")
            _say(explain, line, f"numAdcBits={v[0]} (2=16-bit, the only "
                 f"value in practice). adcOutputFmt={fmt} -> {meaning}."
                 f"{note}")
            p["isComplex"] = fmt != 0
            p["_adcOutputFmt"] = fmt

        elif cmd == "adcbufCfg":
            _say(explain, line,
                 f"subFrameIdx={v[0]} (-1 = applies to all subframes; "
                 "this cfg has only one anyway).\n"
                 f"    adcOutputFmt={v[1]} -> how the on-chip ADC buffer "
                 "stores samples for the internal DSP/HWA path. Keep "
                 "this consistent with adcCfg above.\n"
                 f"    SampleSwap={v[2]} -> I/Q ordering swap, only "
                 "meaningful in complex modes.\n"
                 f"    ChanInterleave={v[3]} -> whether the ADC buffer "
                 "cycles through RX channels first or range samples "
                 "first for a given channel. This project's byte-layout "
                 "assumptions (lesson 05/06) match how this board "
                 "actually streamed real captures - if you change this, "
                 "the reshape logic downstream needs re-deriving.\n"
                 f"    ChirpThreshold={v[4]} -> how many chirps get "
                 "batched into one internal DMA transfer before the DSP "
                 "is interrupted to move them. A performance/latency "
                 "tuning knob, not a data-format change.")

        elif cmd == "profileCfg":
            (profileId, startFreq, idleTime, adcStartTime, rampEndTime,
             txOutPower, txPhaseShifter, freqSlope, txStartTime,
             numAdc, sampleRate, hpf1, hpf2, rxGain) = v[:14]
            p["startFreq_GHz"] = float(startFreq)
            p["idleTime_us"] = float(idleTime)
            p["rampEndTime_us"] = float(rampEndTime)
            p["slope_MHz_us"] = float(freqSlope)
            p["numAdcSamples"] = int(numAdc)
            p["sampleRate_ksps"] = float(sampleRate)
            _say(explain, line,
                 f"profileId={profileId}\n"
                 f"    startFreq={startFreq} GHz - chirp's starting RF "
                 "frequency.\n"
                 f"    idleTime={idleTime} us - dead time between the end "
                 "of one chirp's ramp and the start of the next (lets "
                 "the synthesizer settle back down before ramping "
                 "again).\n"
                 f"    adcStartTime={adcStartTime} us - delay after the "
                 "ramp starts before the ADC begins sampling (skips the "
                 "non-linear start of the ramp).\n"
                 f"    rampEndTime={rampEndTime} us - total duration of "
                 "one frequency ramp (one chirp).\n"
                 f"    txOutPower={txOutPower}, txPhaseShifter="
                 f"{txPhaseShifter} - 0/0 = default output power, no "
                 "extra phase shift.\n"
                 f"    freqSlopeConst={freqSlope} MHz/us - how fast "
                 "frequency ramps. This times the ADC-sampled DURATION "
                 "(not the full ramp) is the bandwidth that actually "
                 "sets your range resolution - see the summary at the "
                 "bottom.\n"
                 f"    txStartTime={txStartTime} us - delay before this "
                 "profile's TX actually starts transmitting, relative to "
                 "the ramp start.\n"
                 f"    numAdcSamples={numAdc} - ADC samples captured PER "
                 "CHIRP PER RX CHANNEL. This is the single biggest lever "
                 "on both range resolution and file size.\n"
                 f"    digOutSampleRate={sampleRate} ksps - the ADC's "
                 "sampling rate.\n"
                 f"    hpfCornerFreq1/2={hpf1}/{hpf2} - high-pass filter "
                 "corner selectors on the receive chain (filters out "
                 "very-low-beat-frequency clutter, i.e. very close/"
                 "strong reflections and DC offset).\n"
                 f"    rxGain={rxGain} - the receive chain's baseband "
                 "gain in dB (this cfg's value encodes gain in the "
                 "chip's internal units, not a plain dB integer - check "
                 "your board's datasheet if you need the exact dB).")

        elif cmd == "chirpCfg":
            chirp_idx, tx_mask = int(v[0]), int(v[7])
            p["chirpTx"][chirp_idx] = tx_mask
            tx_id = tx_mask.bit_length() - 1 if tx_mask else -1
            _say(explain, line,
                 f"This defines ONE entry in the per-loop chirp sequence "
                 f"(chirpStartIdx={v[0]}, chirpEndIdx={v[1]} - a single "
                 "index here since start==end).\n"
                 f"    profileId={v[2]} - which profileCfg to use (we "
                 "only defined one, profile 0).\n"
                 f"    freq/slope/idleTime/adcStartTime VARIATIONS "
                 f"({v[3]}, {v[4]}, {v[5]}, {v[6]}) - per-chirp tweaks on "
                 "top of the profile's base values; 0 here means 'use "
                 "the profile exactly as defined'.\n"
                 f"    txEnable={tx_mask} (binary "
                 f"0b{tx_mask:04b}) -> fires physical TX{tx_id}.\n"
                 f"    So: firing SLOT {chirp_idx} in the loop transmits "
                 f"on physical TX{tx_id}.")

        elif cmd == "frameCfg":
            p["chirpStartIdx"], p["chirpEndIdx"] = int(v[0]), int(v[1])
            p["numLoops"], p["numFrames"] = int(v[2]), int(v[3])
            p["framePeriod_ms"] = float(v[4])
            extra_note = ""
            if len(v) > 7:
                extra_note = (
                    f"\n    NOTE: this line has {len(v)} fields. TI's "
                    "published mmWave SDK reference lists frameCfg as "
                    "exactly 7 fields (chirpStart, chirpEnd, numLoops, "
                    "numFrames, framePeriodicity_ms, triggerSelect, "
                    "frameTriggerDelay_ms) - this device/SDK build has "
                    "one MORE field than that, and I do not have a "
                    "confirmed source for what it is. Rather than guess, "
                    "the honest answer is: ask the firmware. Most TI "
                    "demo CLIs will echo a usage string if you send the "
                    "command name with the WRONG number of args, or "
                    "respond to a bare 'help' over the CLI port with a "
                    "full command list. See the bottom of this lesson "
                    "for a snippet that tries exactly that. The fields "
                    "this project's code actually RELIES on - chirpIdx "
                    "range, numLoops, numFrames, framePeriod - are all "
                    "in the first 5 positions and have been validated "
                    "against real captures throughout this whole "
                    "project, so this ambiguity does not affect anything "
                    "downstream of this lesson.")
            _say(explain, line,
                 f"chirpStartIdx={v[0]}, chirpEndIdx={v[1]} -> which "
                 "chirpCfg entries fire, in order, per loop (here: all "
                 "4, indices 0-3).\n"
                 f"    numLoops={v[2]} -> how many times that whole "
                 "chirp sequence repeats within ONE frame. This is your "
                 "slow-time/Doppler axis length.\n"
                 f"    numFrames={v[3]} -> total frames this capture "
                 "runs for, then stops on its own (0 would mean run "
                 "forever until sensorStop).\n"
                 f"    framePeriodicity={v[4]} ms -> time budgeted "
                 f"between the START of one frame and the next.{extra_note}")

        elif cmd == "lvdsStreamCfg":
            data_fmt = {0: "disabled", 1: "raw ADC data",
                        2: "CP + ADC", 3: "CP + ADC + CQ"}.get(int(v[2]), "?")
            _say(explain, line,
                 f"subFrameIdx={v[0]} (-1 = all).\n"
                 f"    enableHeader={v[1]} -> whether the chip inserts "
                 "its own HSI header per chirp on the LVDS wire itself "
                 "(separate from the DCA1000's own 10-byte UDP packet "
                 "header you'll meet in lesson 03/06 - two different "
                 "headers at two different layers).\n"
                 f"    dataFmt={v[2]} -> {data_fmt}. THIS is the switch "
                 "that makes raw-capture mode possible at all - without "
                 "'1' here, nothing meaningful reaches the DCA1000 "
                 "regardless of anything else in this file.\n"
                 f"    enableSW={v[3]} -> software-triggered streaming "
                 "toggle, left at the known-working default (0) "
                 "throughout this project.")

        elif cmd == "sensorStart":
            _say(explain, line, "The chip actually starts transmitting "
                 "chirps NOW. Everything above this line was just "
                 "accumulating configuration in the chip's memory.")

        elif cmd in ("guiMonitor", "cfarCfg", "multiObjBeamForming",
                      "calibDcRangeSig", "clutterRemoval",
                      "compRangeBiasAndRxChanPhase",
                      "measureRangeBiasAndRxChanPhase", "aoaFovCfg",
                      "cfarFovCfg", "extendedMaxVelocity", "calibData",
                      "lowPower"):
            _say(explain, line, "Configures the chip's ON-CHIP detection "
                 "chain (its own internal range/Doppler/CFAR/angle "
                 "pipeline, used for the point-cloud-over-UART mode). "
                 "This whole tutorial bypasses that entirely - we pull "
                 "RAW samples via the DCA1000 and do our own DSP "
                 "(lessons 07/08) - so these lines are harmless to leave "
                 "at their defaults, but don't affect anything you'll "
                 "read from the DCA1000's raw stream.")

        elif cmd == "antGeometryCfg":
            _say(explain, line, "Describes the PHYSICAL layout of the "
                 "antenna array (which TX/RX pairs form which virtual "
                 "element, and their spacing in wavelengths) so the "
                 "chip's OWN on-chip angle estimator knows the real "
                 "geometry. Lesson 06/08 parse and use this same "
                 "information for our own off-chip angle estimation.")

    p["numTx"] = len({m for m in p["chirpTx"].values() if m}) or 1
    chirps_per_loop = p["chirpEndIdx"] - p["chirpStartIdx"] + 1
    p["chirpsPerFrame"] = chirps_per_loop * p["numLoops"]
    return p


def _say(explain, line, text):
    if not explain:
        return
    print(f"\n{line}")
    print(f"    {text}")


# =======================================================================
# DERIVED RF PERFORMANCE - the numbers that actually answer "what does
# this cfg get me", the way a methodology section in your thesis would
# state them.
# =======================================================================
def summarize_performance(p):
    N, fs = p["numAdcSamples"], p["sampleRate_ksps"] * 1e3
    slope = p["slope_MHz_us"] * 1e12          # MHz/us -> Hz/s
    f0 = p["startFreq_GHz"] * 1e9
    lam = C / f0

    # Bandwidth actually swept WHILE the ADC is sampling (not the whole
    # ramp - adcStartTime skips the non-linear start, and the ramp may
    # run longer than numAdcSamples/fs needs). This, not the full ramp
    # bandwidth, is what sets range resolution.
    adc_sample_duration_s = N / fs
    B_sampled = slope * adc_sample_duration_s

    range_res_m = C / (2 * B_sampled)
    # REAL sampling only gives you N/2 independent range bins for N raw
    # samples (Hermitian symmetry - see lesson 05); a complex-sampled
    # system would get N independent bins for the same N samples. Same
    # per-bin resolution either way, but real sampling needs 2x the raw
    # samples to reach the same MAXIMUM unambiguous range.
    n_usable_bins = N // 2 if not p.get("isComplex", False) else N
    max_range_m = range_res_m * n_usable_bins

    chirp_period_s = (p["idleTime_us"] + p["rampEndTime_us"]) * 1e-6
    tx_count = p["numTx"]
    # TDM-MIMO: each physical TX only fires once every `tx_count` chirps,
    # so the EFFECTIVE pulse interval for Doppler, per TX, is tx_count
    # times the raw chirp period. See lesson 06 for the full picture.
    doppler_pri_s = tx_count * chirp_period_s
    max_velocity_mps = lam / (4 * doppler_pri_s)
    velocity_res_mps = lam / (2 * p["numLoops"] * doppler_pri_s)

    frame_time_ms = p["chirpsPerFrame"] / tx_count * doppler_pri_s * 1e3 \
        if False else p["numLoops"] * doppler_pri_s * 1e3

    print("\n" + "=" * 70)
    print("DERIVED RF PERFORMANCE (the numbers that go in your thesis)")
    print("=" * 70)
    print(f"Center-ish frequency f0          : {f0/1e9:.2f} GHz "
          f"(lambda = {lam*1e3:.3f} mm)")
    print(f"ADC-sampled bandwidth             : {B_sampled/1e9:.3f} GHz")
    print(f"Range resolution                  : {range_res_m*100:.2f} cm/bin")
    print(f"Usable range bins ({'real' if not p.get('isComplex') else 'complex'} "
          f"sampling, N={N})  : {n_usable_bins}")
    print(f"Max unambiguous range             : {max_range_m:.2f} m")
    print(f"Chirp period (idle+ramp)          : {chirp_period_s*1e6:.1f} us")
    print(f"TX count (TDM-MIMO)                : {tx_count}")
    print(f"Effective per-TX Doppler PRI       : {doppler_pri_s*1e3:.4f} ms")
    print(f"Max unambiguous velocity           : +/-{max_velocity_mps:.3f} m/s")
    print(f"Velocity resolution                : {velocity_res_mps*1000:.1f} mm/s "
          f"(over {p['numLoops']} loops)")
    print(f"Frame period (configured)          : {p['framePeriod_ms']:.1f} ms")
    print(f"Frames in this capture             : {p['numFrames']}")
    print(f"Virtual channels (numTx x numRx)   : {p['numTx']} x {p['numRx']} "
          f"= {p['numTx']*p['numRx']}")


def ask_firmware_for_usage(cli_port, cmd_name="frameCfg", baud=115200):
    """When a .cfg has a device/SDK-specific field a published doc
    doesn't cover (like this frameCfg's extra field), the AUTHORITATIVE
    source is the firmware itself. Most TI demo CLIs will print a usage
    string if you send a command with too few/many args, or respond to
    a bare 'help'. This needs REAL hardware - it's here as a template
    for your next lab session, not something demo mode can fake
    meaningfully (we'd just be faking our own guess back to ourselves).
    """
    import serial
    with serial.Serial(cli_port, baud, timeout=0.5) as ser:
        for probe in (cmd_name, "help", f"help {cmd_name}"):
            ser.write((probe + "\n").encode())
            import time
            time.sleep(0.3)
            reply = ser.read(2000).decode(errors="ignore")
            print(f"\n> {probe}\n{reply}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="example_awr2944P.cfg")
    ap.add_argument("--quiet", action="store_true",
                     help="Skip the per-line explanations, just show the "
                          "performance summary.")
    ap.add_argument("--demo", action="store_true",
                     help="(default - this lesson never needs hardware "
                          "unless --ask-firmware is given)")
    ap.add_argument("--ask-firmware", metavar="COM_PORT", default=None,
                     help="Real hardware only: probe the CLI for the "
                          "exact meaning of the undocumented frameCfg "
                          "field (see the note that prints for that "
                          "line).")
    args = ap.parse_args()

    p = parse_cfg_verbose(args.cfg, explain=not args.quiet)
    summarize_performance(p)

    if args.ask_firmware:
        ask_firmware_for_usage(args.ask_firmware)

    print("\nNext: 02_uart_cli_protocol.py - HOW these lines actually get "
          "sent, and the exact timing bug that cost this project a lot "
          "of debugging time.")
