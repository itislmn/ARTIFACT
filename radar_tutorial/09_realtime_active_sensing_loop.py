r"""
LESSON 09 - The acquire / process / decide / act loop ("active sensing")

    python 09_realtime_active_sensing_loop.py

Runs a real (fast) demo loop for a few seconds, no hardware needed.

------------------------------------------------------------------------
"ACTIVE SENSING", CONCRETELY
------------------------------------------------------------------------
Everything through lesson 08 was OFFLINE: capture a fixed block of data,
then process it. "Active" sensing/ML means the loop's own output feeds
back into what happens next - not just "print a number" but "decide
something and DO it", where that action might change the scene itself
(move a servo/rail, redirect a camera, change the radar's own next
configuration - narrower FOV around a target of interest, a different
profile for close range vs far range, etc). That feedback path is the
entire difference between "a script that processes a recording" and
"an active sensing system".

------------------------------------------------------------------------
THE SHAPE OF THE LOOP
------------------------------------------------------------------------
    [capture thread]  --queue-->  [process thread]  -->  [decide + act]
     grabs one frame                runs lessons          state machine;
     worth of raw data              07/08's pipeline       may trigger an
     as fast as the radar           on it                  external action,
     produces it                                           which may in
                                                             turn change what
                                                             gets captured
                                                             next

Two threads, not one, because capture and processing have DIFFERENT time
budgets. The radar keeps producing frames at framePeriod_ms (lesson 01)
regardless of whether your processing code is ready; if you process
in-line in the same loop that reads the socket, a slow processing step
means you fall behind and start dropping/missing frames (or worse,
reading a stale/partial frame). A queue between the two lets the
capture side always keep up with the hardware, and lets you explicitly
see and handle the case where processing genuinely can't keep up
(the queue grows - lesson tracks and reports that explicitly below,
rather than silently falling behind).

------------------------------------------------------------------------
THE REAL FRAME-RATE BUDGET
------------------------------------------------------------------------
This project's real .cfg (lesson 01) has framePeriod_ms ~656ms - about
1.5 frames/second. Your processing pipeline (range FFT + Doppler FFT +
CFAR + angle estimate, lessons 07/08) needs to finish well within that
budget, on average, or the queue backs up forever. This lesson's demo
uses a much faster synthetic frame rate (20ms, ~50 fps) specifically so
you can watch the queue-depth reporting actually mean something in a
few seconds of runtime instead of needing to wait minutes.
"""
import queue
import threading
import time

import numpy as np


# =======================================================================
# STAGE 1 - capture (a stand-in for lesson 04's real UDP receive loop;
# here just a timer producing a synthetic "detection" at a slowly
# changing angle, standing in for a target walking past the radar)
# =======================================================================
def capture_thread_fn(out_queue, stop_event, frame_period_s=0.02):
    frame_id = 0
    t0 = time.time()
    while not stop_event.is_set():
        t_frame = time.time()
        # Stand-in "raw frame": in reality this is lesson 04's UDP
        # receive + lesson 06's cube reshape. Here it's just enough
        # synthetic ground truth for stage 2 to have something to find -
        # a target sweeping from -40 to +40 degrees over ~4 seconds, at
        # a range that breathes a little (simulating someone walking).
        elapsed = t_frame - t0
        true_angle = -40 + 20 * elapsed
        true_range = 1.5 + 0.1 * np.sin(elapsed * 2)
        frame = {"frame_id": frame_id, "t": t_frame,
                 "true_angle": true_angle, "true_range": true_range}
        try:
            out_queue.put_nowait(frame)
        except queue.Full:
            pass  # see the note on backpressure in run_loop() below
        frame_id += 1
        # Keep a steady cadence regardless of how long this iteration's
        # own bookkeeping took - the real DCA1000 doesn't wait for you.
        sleep_left = frame_period_s - (time.time() - t_frame)
        if sleep_left > 0:
            time.sleep(sleep_left)


# =======================================================================
# STAGE 2 - process (stands in for lessons 07/08's range/Doppler/CFAR/
# angle pipeline - kept lightweight here since the POINT of this lesson
# is the loop structure, not re-deriving DSP already proven elsewhere)
# =======================================================================
def process_frame(frame, rng):
    # Simulate realistic measurement noise on top of the "true" values -
    # a real pipeline's precision here comes directly from lessons 07/08.
    measured_angle = frame["true_angle"] + rng.normal(0, 0.5)
    measured_range = frame["true_range"] + rng.normal(0, 0.02)
    # Simulate the pipeline actually taking measurable time, like a real
    # FFT-based one would - small but non-zero, so queue depth is
    # meaningful in this demo instead of trivially always zero.
    time.sleep(0.005)
    return {"frame_id": frame["frame_id"], "angle": measured_angle,
            "range": measured_range}


# =======================================================================
# STAGE 3 - decide + act: the part that makes it "active". A tiny state
# machine with hysteresis (don't act on every single noisy frame - only
# when the target has been in the "trigger zone" for a few consecutive
# frames), and an action hook that's just a function call here but is
# exactly where you'd put a serial/GPIO command to a rail, a servo, a
# camera gimbal, or a re-configuration of the radar itself.
# =======================================================================
class TrackAndAct:
    def __init__(self, trigger_angle_deg=0.0, tolerance_deg=5.0,
                 confirm_frames=3, on_action=None):
        self.trigger_angle_deg = trigger_angle_deg
        self.tolerance_deg = tolerance_deg
        self.confirm_frames = confirm_frames
        self.on_action = on_action or (lambda info: None)
        self._consecutive_in_zone = 0
        self._last_action_frame = -999
        self._cooldown_frames = 50   # don't re-trigger every single frame

    def update(self, measurement):
        in_zone = abs(measurement["angle"] - self.trigger_angle_deg) < self.tolerance_deg
        self._consecutive_in_zone = self._consecutive_in_zone + 1 if in_zone else 0

        ready = self._consecutive_in_zone >= self.confirm_frames
        cooled_down = (measurement["frame_id"] - self._last_action_frame) > self._cooldown_frames
        if ready and cooled_down:
            self._last_action_frame = measurement["frame_id"]
            self.on_action(measurement)
            return True
        return False


def example_action(measurement):
    """Where a real system would issue a command - move a rail to track
    this bearing, snap a camera to this angle, switch the radar to a
    narrow high-resolution profile centered on this range. Here: just a
    clearly-labeled log line, so the HOOK is obvious without hardware to
    actually drive."""
    print(f"  >>> ACTION: center on bearing {measurement['angle']:+.1f} deg, "
          f"range {measurement['range']:.2f} m (frame {measurement['frame_id']})")


# =======================================================================
# THE LOOP ITSELF
# =======================================================================
def run_loop(duration_s=4.0):
    frame_queue = queue.Queue(maxsize=200)
    stop_event = threading.Event()
    rng = np.random.default_rng(0)
    tracker = TrackAndAct(trigger_angle_deg=0.0, tolerance_deg=5.0,
                           confirm_frames=3, on_action=example_action)

    cap_thread = threading.Thread(target=capture_thread_fn,
                                   args=(frame_queue, stop_event), daemon=True)
    cap_thread.start()

    print(f"Running for {duration_s:.0f}s (synthetic target sweeping -40 "
          "to +40 deg, action triggers within +/-5 deg of boresight)...\n")
    t0 = time.time()
    n_processed = 0
    max_queue_depth = 0
    while time.time() - t0 < duration_s:
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        max_queue_depth = max(max_queue_depth, frame_queue.qsize())
        measurement = process_frame(frame, rng)
        tracker.update(measurement)
        n_processed += 1

    stop_event.set()
    cap_thread.join(timeout=1.0)

    print(f"\nProcessed {n_processed} frames in {duration_s:.0f}s "
          f"({n_processed/duration_s:.1f} fps).")
    print(f"Max queue depth seen: {max_queue_depth} "
          f"({'kept up fine' if max_queue_depth < 5 else 'processing was falling behind - see the note below'}).")
    if max_queue_depth >= 5:
        print("A growing queue depth means your processing stage is slower")
        print("than your capture rate on average - the fix is almost never")
        print("'add a bigger queue' (that just adds latency until it")
        print("overflows anyway); it's speeding up the processing stage,")
        print("or accepting a lower effective frame rate on purpose.")


if __name__ == "__main__":
    run_loop()
    print("\nNext: 10_end_to_end_capstone.py - lessons 01-08, once each, in")
    print("one compact script: configure, capture, process, detect,")
    print("estimate angle, print result.")
