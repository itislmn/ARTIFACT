r"""
LESSON 00 - Hardware overview: what each board does and how they talk

    python 00_hardware_overview.py            (safe to run anytime, no hardware needed)

Read this file top to bottom before touching anything else in this folder.
Every later lesson assumes you understand the picture drawn here.

------------------------------------------------------------------------
THE TWO BOARDS, AND WHY YOU NEED BOTH
------------------------------------------------------------------------

    [ AWR2944P EVM ]  --LVDS (raw ADC samples)-->  [ DCA1000EVM ]
          |                                              |
          | UART (2 COM ports)                           | Ethernet (UDP)
          v                                              v
    [ your laptop, CLI/config channel ]        [ your laptop, data channel ]

AWR2944P EVM - the actual radar. It has:
  - The RF front end: 4 TX antennas, 4 RX antennas, all on-board, all
    aimed the same direction (see lesson 06 for why 4x4 -> 16 "virtual"
    channels).
  - A DSP/MCU (the "mmWave demo" firmware you flash onto it) that
    generates chirps, digitizes the received signal, and can ALSO do
    on-chip detection (range/Doppler/CFAR/angle) and report just a
    point cloud over UART if you don't need the raw samples.
  - Two USB-to-UART bridges, which is why plugging it in gives you TWO
    COM ports: one is the "CLI port" (you type config commands into it,
    115200 baud, plain ASCII), the other is the "data port" (binary
    point-cloud output IF you're using the on-chip detector instead of
    raw capture - this whole tutorial mostly bypasses it in favor of
    the DCA1000's raw stream).

DCA1000EVM - a data-capture accessory, nothing more. It has NO radar
  logic of its own. Its only job: sit on the AWR2944P's LVDS pins,
  capture whatever raw ADC samples the chip streams out, and re-package
  them as UDP/Ethernet packets your laptop can receive. You configure
  IT (not the radar) with a small binary command protocol over UDP -
  see lesson 03. It also has its own 60-pin high-speed connector to the
  AWR2944P, plus jumpers you set once and generally never touch again.

Why raw capture needs BOTH boards talking, in sync, at once: the radar
only streams samples out over LVDS while it's actively running a frame
sequence (i.e. after you send "sensorStart" over UART); the DCA1000 only
forwards what arrives on those LVDS pins while it's armed and recording
(after you send it RECORD_START over UDP). Get the order or timing wrong
and you capture nothing - see lesson 04 for exactly how that bit us.

------------------------------------------------------------------------
JUMPERS AND SWITCHES (set once, physically, before anything below works)
------------------------------------------------------------------------
AWR2944P EVM:
  - SOP (Sense-On-Power) jumpers select boot mode. For flashing new
    firmware you need "flash programming mode"; for normal running you
    need "functional mode". Check your EVM's silkscreen/manual for the
    exact jumper positions - they differ by board revision, and getting
    this wrong is the #1 reason uart_uniflash.py hangs waiting for "C"
    bytes that never come.

DCA1000EVM:
  - SW1 config switch: SW_CONFIG (PC controls it over Ethernet - what
    this whole tutorial assumes) vs FUNC_CONFIG (standalone/switch-only
    mode). Must be SW_CONFIG.
  - A 5V barrel power input, separate from the AWR2944P's own power -
    both boards need their own power, and the DCA1000 will simply not
    respond to anything if it isn't powered.

------------------------------------------------------------------------
NETWORKING (also set once)
------------------------------------------------------------------------
The DCA1000 has a fixed factory IP: 192.168.33.180. Your laptop's
Ethernet adapter (the one physically cabled to the DCA1000) must be
given the STATIC IP 192.168.33.30 - not DHCP, a fixed address you set
by hand in your OS network settings - or nothing on the UDP side of
this tutorial will connect. This is a common silent failure: the code
will look like it's "hanging" on a socket bind or timing out on every
command with no other symptom.

------------------------------------------------------------------------
WHAT THIS TUTORIAL SERIES COVERS, IN ORDER
------------------------------------------------------------------------
  00 (this file)  - the picture above
  01 - .cfg file  - every line of the config you send the radar, and
                     the RF math (range/velocity resolution etc.) each
                     one controls
  02 - UART       - the CLI protocol that sends that .cfg file, and its
                     "Done"/error reply pattern
  03 - DCA1000 UDP- the binary command protocol that arms the capture
                     card
  04 - orchestration - running UART config-send and UDP receive
                     CONCURRENTLY, and the exact timing bug this project
                     hit when that wasn't done right
  05 - ADC format - what a "raw sample" physically is, real vs complex,
                     why this chip family only supports real
  06 - radar cube - reshaping the raw byte stream into a
                     (frame, chirp, virtual-channel, sample) array, and
                     TDM-MIMO: why cycling 4 TX antennas fakes a bigger
                     antenna array
  07 - range/Doppler/CFAR - the actual signal processing
  08 - angle estimation - turning phase differences across the virtual
                     array into a bearing
  09 - real-time active sensing loop - the acquire/process/decide
                     structure for closing a loop around what you sense
  10 - capstone   - all of the above, combined, in one compact script

Every lesson from 02 onward can run in "--demo" mode with ZERO hardware
attached (it fakes the serial port and the DCA1000 over UDP, see
common.py), so you can read AND RUN every concept before you're ever
back in the lab. Swap in --cli COM4 (and, from lesson 03 on, real
network access to 192.168.33.180) to point the same code at the real
boards.
"""
import sys


def find_com_ports():
    """REAL, runnable: lists every serial port Windows/macOS/Linux
    currently sees, so you can tell which two belong to the AWR2944P.
    On Windows they'll show as "COM3", "COM4", ...; on Linux/macOS as
    /dev/ttyUSB0, /dev/ttyACM0, etc. The AWR2944P's pair is usually
    adjacent port numbers that appear/disappear together when you
    plug/unplug the board - if you're not sure which two are "yours",
    unplug the board, re-run this, plug it back in, re-run again, and
    diff the lists.
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        print("pyserial isn't installed here. pip install pyserial to run "
              "this check for real; skipping.")
        return []
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports detected at all right now.")
        return []
    print(f"{len(ports)} serial port(s) visible to this machine:")
    for p in ports:
        print(f"  {p.device:10s}  {p.description}")
    return [p.device for p in ports]


def check_network_interface():
    """REAL, runnable, best-effort: looks for a network interface holding
    the 192.168.33.30 address the DCA1000 side of this tutorial needs.
    Not able to fix it for you (that's an OS network-settings change),
    but tells you immediately whether it's already set, which is faster
    than debugging a mysterious socket timeout three lessons from now.
    """
    import socket
    try:
        # Doesn't actually send anything - UDP connect() just asks the OS
        # "which local address would you use to reach this remote one",
        # which is a quick way to see our outbound-facing address.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("192.168.33.180", 4096))
        local_ip = s.getsockname()[0]
        s.close()
        if local_ip == "192.168.33.30":
            print(f"OK: this machine would talk to the DCA1000 from "
                  f"{local_ip} - that's the expected static IP.")
        else:
            print(f"WARNING: this machine would talk to the DCA1000 from "
                  f"{local_ip}, not 192.168.33.30. Set your Ethernet "
                  "adapter's IPv4 address to a static 192.168.33.30 "
                  "before lesson 03/04.")
    except OSError as e:
        print(f"Could not even attempt a route to 192.168.33.180: {e}")
        print("Likely no interface on the 192.168.33.x subnet exists yet - "
              "set one up (static IP 192.168.33.30) before lesson 03/04.")


if __name__ == "__main__":
    print(__doc__)
    print("=" * 70)
    print("LIVE CHECKS (safe - read-only, no commands sent to any board)")
    print("=" * 70)
    find_com_ports()
    print()
    check_network_interface()
    print()
    print("Next: read and run 01_cfg_file_reference.py")
    sys.exit(0)
