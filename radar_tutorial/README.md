# radar_tutorial — learn your AWR2944P + DCA1000 setup, one idea at a time

Each file below is a standalone lesson. Open it, read the comments top to
bottom BEFORE running it — they're written like I'm sitting next to you
explaining what you're about to see. Then run it. Then go look at what it
saved in `output/<lesson_name>/`.

Run everything from inside this folder. Most scripts need `--cli` and/or
`--cfg` arguments pointing at your COM port and your sensor config file —
run any script with `--help` to see exactly what it wants.

## The order, and why

| # | File | Needs hardware? | Teaches |
|---|---|---|---|
| 01 | `01_check_hardware_alive.py` | Yes | Is anything even connected and responding |
| 02 | `02_understand_your_cfg.py` | No | What every line in your `.cfg` actually means |
| 03 | `03_get_a_pointcloud.py` | Yes | The easy path: chip does the DSP, you get points |
| 04 | `04_simulate_a_chirp_no_hardware.py` | No | Chirp → beat signal → FFT → range, with fake data you can trust |
| 05 | `05_dca1000_protocol_explained.py` | Yes | The raw UDP bytes that configure the DCA1000, one at a time |
| 06 | `06_capture_raw_iq.py` | Yes | Combining 02+05 into one real raw-ADC capture |
| 07 | `07_build_and_explore_cube.py` | No (uses 06's file) | Turning that file into a labeled 4D array |
| 08 | `08_range_fft_from_scratch.py` | No (uses 07's file) | The real range FFT, on your real data |
| 09 | `09_doppler_and_range_doppler_map.py` | No (uses 07's file) | The second FFT: velocity |
| 10 | `10_cfar_detection.py` | No (uses 09's file) | Automatically finding real targets in the map |
| 11 | `11_realtime_loop_skeleton.py` | Yes | Putting 06+07+08+09+10 into one live loop — the seed of active sensing |

**The honest ugly bits**, called out explicitly rather than hidden:
- Lesson 05's exact packet-header byte layout is community-reverse-engineered,
  not from an official spec I could verify — it worked on your hardware, but
  say so if you ever present it as gospel.
- Lesson 09 will very likely show you a real, known TDM-MIMO artifact
  (energy pinned at the edges of the velocity axis) rather than a clean
  moving target. That's explained in the file itself, not swept under the rug.
