"""XP12 — load-adaptive power governor.

A control loop that watches the live request rate and moves the board between
15 W / 25 W / MAXN_SUPER to spend the least energy while defending the latency SLA.

Policy is deliberately asymmetric: **scale up immediately** when demand rises (a
missed SLA is worse than a few extra watts), but **scale down lazily** (hysteresis
band + minimum dwell time) so it doesn't flap around a threshold or pay the ~1–2 s
mode-switch cost repeatedly. Thresholds come from the measured SLA-safe capacities
(XP9/XP11): 15 W ≈ 300, 25 W ≈ 460, MAXN ≈ 510 req/s.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))

MODES = {0: "15W", 1: "25W", 2: "MAXN_SUPER"}


def set_mode(m: int) -> None:
    subprocess.run(f"echo a | sudo -S nvpmodel -m {m}", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run("echo a | sudo -S jetson_clocks", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Governor(threading.Thread):
    def __init__(self, server, interval=1.0, min_dwell=5.0, start_mode=2):
        super().__init__(daemon=True)
        self.server = server
        self.interval = interval
        self.min_dwell = min_dwell
        self.mode = start_mode
        self.stop_evt = threading.Event()
        self.timeline = []            # (t_rel, mode, measured_rps)
        self.t0 = None

    def _choose_mode(self, rps: float) -> int:
        # up thresholds (scale up to meet demand) and lower down thresholds (hysteresis)
        up = 2 if rps > 400 else 1 if rps > 210 else 0
        dn = 2 if rps > 340 else 1 if rps > 150 else 0
        if up > self.mode:
            return up
        if dn < self.mode:
            return dn
        return self.mode

    def run(self):
        set_mode(self.mode)
        self.t0 = time.perf_counter()
        last_c, last_t, last_switch = self.server.submitted, self.t0, self.t0
        while not self.stop_evt.is_set():
            time.sleep(self.interval)
            now, c = time.perf_counter(), self.server.submitted
            rps = (c - last_c) / max(1e-6, now - last_t)
            last_c, last_t = c, now
            tgt = self._choose_mode(rps)
            if tgt > self.mode:                              # scale UP now
                set_mode(tgt); self.mode = tgt; last_switch = now
            elif tgt < self.mode and now - last_switch >= self.min_dwell:  # scale DOWN lazily
                set_mode(tgt); self.mode = tgt; last_switch = now
            self.timeline.append((round(now - self.t0, 1), self.mode, round(rps)))

    def stop(self):
        self.stop_evt.set()
