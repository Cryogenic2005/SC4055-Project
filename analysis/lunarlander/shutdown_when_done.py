"""Shut the machine down when the LunarLander sweep finishes — and only then.

Waits for all six arms to land in rows/, gives the driver a few minutes to write
the summary, then schedules `shutdown /s` with a 3-minute delay so it can still be
cancelled with `shutdown /a`.

Two guards, because a shutdown is not undoable:
  * a hard deadline (default 22 h) — if the sweep has not finished by then, log it
    and exit WITHOUT shutting down, so a stuck job never costs you the machine;
  * the shutdown is only armed when rows/ is complete, never on process death.

Run detached:
    Start-Process .venv\\Scripts\\python.exe -ArgumentList "-u","shutdown_when_done.py"
"""

from __future__ import annotations

import datetime as dt
import subprocess
import time
from pathlib import Path

SWEEP_DIR = Path(r"C:\Users\Mingchen\SC4055-Project\analysis_out\lunarlander_sweep")
LOG = SWEEP_DIR / "shutdown.log"
TOTAL_ARMS = 6
GRACE_MIN = 3          # let the driver write the summary first
DEADLINE_H = 22
POLL_S = 120
SHUTDOWN_DELAY_S = 180  # `shutdown /a` still works during this window


def note(message: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp}  {message}\n")
    print(f"{stamp}  {message}", flush=True)


def arms_done() -> int:
    return len(list((SWEEP_DIR / "rows").glob("*.json")))


def main() -> None:
    started = time.time()
    note(f"shutdown watcher armed: will power off after {TOTAL_ARMS} arms "
         f"(deadline {DEADLINE_H} h)")

    while True:
        done = arms_done()
        if done >= TOTAL_ARMS:
            note(f"{done}/{TOTAL_ARMS} arms done; waiting {GRACE_MIN} min for the summary")
            time.sleep(GRACE_MIN * 60)
            note(f"issuing shutdown /s /t {SHUTDOWN_DELAY_S} — cancel with 'shutdown /a'")
            subprocess.run(["shutdown", "/s", "/t", str(SHUTDOWN_DELAY_S),
                            "/c", "LunarLander sweep finished"], check=False)
            return

        elapsed_h = (time.time() - started) / 3600
        if elapsed_h > DEADLINE_H:
            note(f"deadline reached with {done}/{TOTAL_ARMS} arms — NOT shutting down")
            return

        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
