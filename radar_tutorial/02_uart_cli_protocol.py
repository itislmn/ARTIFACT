r"""
LESSON 02 - The UART CLI protocol, and a real timing bug you'll hit

    python 02_uart_cli_protocol.py --demo           (fake serial port, always works)
    python 02_uart_cli_protocol.py --cli COM4        (real hardware)

------------------------------------------------------------------------
THE PROTOCOL ITSELF IS SIMPLE
------------------------------------------------------------------------
It's a plain ASCII, line-oriented, request/reply protocol over a normal
serial port at 115200 baud, 8N1. You write a line of text ending in
"\n"; the firmware parses it, applies it, and writes back either:

    <your command echoed>\r\n
    Done\r\n
    mmwDemo:/>

...on success, or an error message instead of "Done" on failure. There
is no binary framing, no checksum, no length prefix - it's exactly like
typing into any interactive serial console (because that's what it is;
you could do this whole lesson by hand in a terminal emulator like
Tera Term or PuTTY, one line at a time).

The whole .cfg file from lesson 01 is just a batch of these lines, sent
in order, with each one's reply checked before sending the next. If any
line comes back with anything other than "Done", the chip's
configuration state is now UNKNOWN - stop, don't push forward, is the
right instinct (this project's code always aborts immediately on that,
see `strict=True` below).

------------------------------------------------------------------------
THE PART THAT ISN'T SIMPLE: HOW LONG DO YOU WAIT FOR A REPLY?
------------------------------------------------------------------------
This sounds like a non-question until you hit it. Two wrong answers,
both tempting:

  WRONG #1: ser.read(400) with a SHORT port timeout (e.g. 0.05s). You'll
  frequently read only PART of the reply (whatever arrived in the first
  50ms) and miss "Done" arriving 10ms later, making a successful command
  look like a failure.

  WRONG #2 (what an earlier version of this project's code actually
  did): ser.read(400) with a LONG port timeout (e.g. 1 second). This
  looks safe - surely you'll never miss a reply if you wait a whole
  second? But pyserial's read(n) does NOT return as soon as some data
  arrives. It keeps trying to fill all 400 bytes and only gives up once
  the FULL configured timeout elapses. A 5-byte "Done\r\n" reply that
  physically arrives in 8 milliseconds still makes read(400) block for
  the entire 1000ms before returning. With ~30 lines in a real .cfg,
  that turned "send the config" into a ~30 SECOND operation - and
  because the DCA1000's receive window (lesson 04) only stays open for
  a few seconds, the window had already closed before sensorStart was
  even transmitted. The result: "0 packets received", every time,
  with nothing in the logs pointing at the real cause. This was a real
  bug in this exact project, and it's demonstrated live below.

  RIGHT: read in a loop with a SHORT per-call port timeout, and decide
  "the reply is finished" based on a QUIET PERIOD (no new bytes for,
  say, 150ms) rather than a fixed total wait. That returns almost
  immediately for a fast reply, and still safely waits out a slow one.
  This is _read_reply() below - watch the timing difference for
  yourself.
"""
import argparse
import time

from common import FakeSerial


def naive_read(ser, nbytes=400):
    """WRONG #2 from above. Included so you can WATCH it be slow, not
    just be told it's slow."""
    return ser.read(nbytes)


def quiet_period_read(ser, quiet_after=0.15, max_wait=2.0):
    """RIGHT. Keep reading in small chunks; stop once nothing NEW has
    arrived for `quiet_after` seconds (or `max_wait` as a hard ceiling
    so a truly dead port doesn't hang forever)."""
    buf = b""
    t0 = time.time()
    last_data_t = t0
    while time.time() - t0 < max_wait:
        chunk = ser.read(256)
        if chunk:
            buf += chunk
            last_data_t = time.time()
        elif time.time() - last_data_t > quiet_after:
            break
    return buf


DEMO_CFG_LINES = [
    "sensorStop", "flushCfg", "dfeDataOutputMode 1", "channelCfg 15 15 0 0 0",
    "adcCfg 2 0", "adcbufCfg -1 1 1 1 1", "profileCfg 0 77 186 7 57.14 0 0 70 1 656 13349 0 0 158",
    "chirpCfg 0 0 0 0 0 0 0 1", "chirpCfg 1 1 0 0 0 0 0 4",
    "chirpCfg 2 2 0 0 0 0 0 8", "chirpCfg 3 3 0 0 0 0 0 2",
    "frameCfg 0 3 64 5 656 100 1 0", "lvdsStreamCfg -1 0 1 0", "sensorStart",
]


def send_config_naive(ser, lines):
    t0 = time.time()
    for line in lines:
        ser.write((line + "\n").encode())
        reply = naive_read(ser)
        ok = b"Done" in reply
        print(f"  {line[:40]:40s} -> {'Done' if ok else 'NO/short REPLY'}")
    return time.time() - t0


def send_config_fixed(ser, lines):
    t0 = time.time()
    for line in lines:
        ser.write((line + "\n").encode())
        reply = quiet_period_read(ser)
        ok = b"Done" in reply.decode(errors="ignore").encode()
        print(f"  {line[:40]:40s} -> {'Done' if ok else 'NO/short REPLY'}")
    return time.time() - t0


def run_demo():
    print("Simulating a firmware that replies in ~10ms per line (realistic")
    print("for this hardware) but with the port's OWN per-call timeout set")
    print("to 1 second, exactly like the buggy version of this project's")
    print("code once did.\n")

    print("--- Using naive_read() (read(400), no quiet-period logic) ---")
    ser = FakeSerial("FAKE", timeout=1.0, reply_delay_s=0.01)
    with ser:
        elapsed_naive = send_config_naive(ser, DEMO_CFG_LINES)
    print(f"Total time for {len(DEMO_CFG_LINES)} lines: {elapsed_naive:.2f}s\n")

    print("--- Using quiet_period_read() (the fix) ---")
    ser2 = FakeSerial("FAKE", timeout=0.05, reply_delay_s=0.01)
    with ser2:
        elapsed_fixed = send_config_fixed(ser2, DEMO_CFG_LINES)
    print(f"Total time for {len(DEMO_CFG_LINES)} lines: {elapsed_fixed:.2f}s\n")

    print(f"Speedup: {elapsed_naive/max(elapsed_fixed,1e-6):.1f}x. Both sent")
    print("the EXACT same 14 lines to the EXACT same (simulated) firmware -")
    print("the only difference is how the code decided a reply was done")
    print("being received. Now imagine the real .cfg's ~30 lines, and a")
    print("DCA1000 receive window that only stays open for a few seconds:")
    print("this is precisely why captures were silently coming back with")
    print("0 packets earlier in this project.")


def run_real(cli_port):
    import serial
    print(f"Sending {len(DEMO_CFG_LINES)} lines to real hardware on {cli_port}...")
    with serial.Serial(cli_port, 115200, timeout=0.05) as ser:
        ser.write(b"\n")
        time.sleep(0.2)
        ser.reset_input_buffer()
        t0 = time.time()
        for line in DEMO_CFG_LINES:
            ser.write((line + "\n").encode())
            reply = quiet_period_read(ser).decode(errors="ignore")
            ok = "Done" in reply
            print(f"  {line[:40]:40s} -> {'Done' if ok else reply.strip()[:50] or 'NO REPLY'}")
        print(f"Total: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=None, help="Real COM port, e.g. COM4")
    ap.add_argument("--demo", action="store_true",
                     help="Run the timing comparison against a fake port "
                          "(default if --cli is not given).")
    args = ap.parse_args()

    if args.cli:
        run_real(args.cli)
    else:
        run_demo()

    print("\nNext: 03_dca1000_udp_protocol.py - the OTHER protocol, a "
          "binary one, that controls the DCA1000 capture card.")
