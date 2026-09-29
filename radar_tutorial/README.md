# radar_tutorial

A from-zero-to-expert path through the AWR2944P EVM + DCA1000EVM setup,
covering **both** ways this hardware gets you data: the raw-ADC/DCA1000
pipeline (the .cfg file, the two control protocols, ADC/IQ data, the radar
cube, TDM-MIMO, range/Doppler/CFAR, angle estimation) AND the chip's own
on-board point-cloud detection over UART (the TLV protocol) - plus saving
and visualizing what you capture, and a real-time active-sensing loop.
Each lesson is one focused, heavily-commented, **runnable** Python file.

## Running these lessons

Every lesson runs standalone with **zero hardware**:

```
python 00_hardware_overview.py
python 01_cfg_file_reference.py
python 02_uart_cli_protocol.py --demo
python 03_dca1000_udp_protocol.py
python 04_capture_orchestration_and_timing.py --demo
python 05_adc_sample_format_and_iq.py
python 06_radar_cube_and_tdm_mimo.py
python 07_range_doppler_cfar.py
python 08_angle_estimation_basics.py
python 09_realtime_active_sensing_loop.py
python 10_end_to_end_capstone.py --demo
python 11_uart_pointcloud_tlv.py --demo
python 12_saving_and_visualizing_the_cube.py --demo
```

Demo mode uses a fake serial port and a fake DCA1000 (`common.py`) that
speak the *exact same wire protocols* as the real hardware - including
reproducing the real timing bug this project hit (lesson 04) and the real
TX-antenna mislabeling bug this project hit (lesson 06), live, with
before/after numbers. Nothing in these demos is hand-waved to "look right";
every lesson that makes a numeric claim ends with an `assert` that actually
checks it, and every one of them passes as shipped.

Lessons 02, 04, 10, and 11 also take real hardware:

```
python 02_uart_cli_protocol.py --cli COM4
python 04_capture_orchestration_and_timing.py --cli COM4
python 10_end_to_end_capstone.py --cli COM4 --cfg example_awr2944P.cfg --seconds 3
python 11_uart_pointcloud_tlv.py --real --cli COM4
```

Lesson 12 never touches hardware at all - it works on any radar cube, real
or synthetic, so its `--demo` also stands in for "run this on your own
saved capture" (swap the synthesized cube for one you loaded from your own
`.npz` or `.bin` file and everything downstream - save, reload, plot - is
identical).

`example_awr2944P.cfg` is this project's own real, working sensor config
(the one lesson 01 parses in detail) - a corrected copy with `adcCfg 2 0`,
matching what this whole project validated against real captures.

## Suggested order

Read and run them in numeric order once - each one builds on the last:

| # | Lesson | Core question it answers |
|---|--------|---------------------------|
| 00 | Hardware overview | What does each board do, and how are they cabled/networked? |
| 01 | .cfg file reference | What does every single line of the config actually mean? |
| 02 | UART CLI protocol | How does the config actually get sent, and what timing trap is there? |
| 03 | DCA1000 UDP protocol | How do you arm the capture card itself? |
| 04 | Capture orchestration | How do the UART and UDP sides run together without racing each other? |
| 05 | ADC/IQ format | What IS a raw sample - and where does "IQ"/phase actually come from? |
| 06 | Radar cube + TDM-MIMO | How do you turn bytes into a (frame, chirp, channel, sample) array, correctly? |
| 07 | Range/Doppler/CFAR | How do you turn that cube into an actual range and velocity? |
| 08 | Angle estimation | How do you turn phase differences across channels into a bearing? |
| 09 | Active sensing loop | How do you structure this as a continuous, closed-loop system? |
| 10 | Capstone | All of the above, once each, start to finish. |
| 11 | UART point-cloud / TLV | How does the chip's own on-board detection stream out, and how do you decode it? |
| 12 | Saving + visualizing the cube | How do you save a capture, reload it later, and actually look at it? |

Lessons 00-10 and 12 are all about the **raw-ADC path**: you get bytes, you
process them yourself (this is what `pattern_measurement/` uses). Lesson 11
is the **other** path this hardware supports: the chip runs its own
detection internally and hands you finished (x, y, z, velocity) points over
plain UART - no DCA1000, no raw cube, much less data, much less control.
Both paths are real, documented features of this hardware; which one you
want depends on whether you need the raw signal (angle-of-arrival research,
custom processing) or just "where are the objects" (lesson 11's path is
far simpler and lower-bandwidth for that).

## A note on accuracy and honesty

This rewrite exists because an earlier version of this tutorial (and this
project's early code) contained real mistakes - most seriously, wrong
guidance on `adcCfg` (complex vs real sampling), a DCA1000 status-byte
misparse, and a UART timing bug that together caused persistent, confusing
"0 packets received" failures. Every one of those is now explained
*as a worked example* in the relevant lesson, with the real bytes/numbers
involved, rather than swept under the rug.

Where I could not find or confirm an authoritative answer (part of
`frameCfg`'s exact field layout on this specific device/SDK; the full,
certified field-by-field decode of `antGeometryCfg` beyond its confirmed
spacing values), the lessons say so explicitly, rather than presenting a
guess with false confidence - and show you how to get the authoritative
answer yourself (asking the firmware directly, or checking the SDK
source), which is a genuinely useful skill beyond just this one project.

## Relationship to your other project folders

- `pattern_measurement/` (antenna/radome pattern sweep tooling) and any
  earlier DOA (direction-of-arrival) project built are the
  **applied, production version** of what lessons 06/08 teach in
  simplified/illustrative form. If you still have that DOA project's
  geometry-parsing code, it's the authoritative source for this exact
  array's real element positions - more authoritative than this
  tutorial's honestly-labeled stand-in geometry.
- `common.py` in this folder is test scaffolding only (a fake serial
  port + fake DCA1000), used so every lesson's demo mode needs no
  hardware. It is not part of the actual capture pipeline.
