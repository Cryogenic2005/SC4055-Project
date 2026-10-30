"""Watchdog for the 16 h LunarLander sweep.

Checks every five minutes that the sweep is still making progress, restarts it if
it died, and writes one line per check to watchdog.log. The sweep is resumable
(rows/*.json), so a restart costs at most the arm that was in flight.

"Making progress" is measured by the freshness of the training CSV, not the log:
the log only prints every 250 episodes, which is over an hour apart under noise.

Run detached:
    Start-Process .venv\\Scripts\\python.exe -ArgumentList "-u", watchdog.py
"""

from __future__ import annotations

import datetime as dt
import subprocess
import time
from pathlib import Path

ROOT = Path(r"C:\Users\Mingchen\SC4055-Project")
SWEEP_DIR = ROOT / "analysis_out" / "lunarlander_sweep"
LOG = SWEEP_DIR / "watchdog.log"
DRIVER = Path(r"C:\Users\Mingchen\AppData\Local\Temp\qnprobe\ll_16h.py")
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

CHECK_EVERY_S = 300
STALE_MIN = 20      # a training CSV untouched this long means the worker is stuck
TOTAL_ARMS = 6      # 3 clean + 3 noise-trained


def python_command_lines() -> list[str]:
    """Command lines of live python processes (WMI; no psutil in this venv)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=60,
        )
        return [ln for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:  # noqa: BLE001 - a failed probe must not kill the watchdog
        return []


def newest_csv_age_min() -> float | None:
    files = list((SWEEP_DIR / "runs" / "train").glob("*.csv"))
    if not files:
        return None
    newest = max(f.stat().st_mtime for f in files)
    return (time.time() - newest) / 60


def finished_arms() -> int:
    return len(list((SWEEP_DIR / "rows").glob("*.json")))


def note(message: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp}  {message}\n")
    print(f"{stamp}  {message}", flush=True)


def start_driver() -> None:
    subprocess.Popen(
        [str(PYTHON), "-u", str(DRIVER)],
        cwd=str(ROOT),
        stdout=open(SWEEP_DIR / "log16h.txt", "a", encoding="utf-8"),
        stderr=open(SWEEP_DIR / "err16h.txt", "a", encoding="utf-8"),
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
    )


def main() -> None:
    note("watchdog started")
    while True:
        arms = finished_arms()
        if arms >= TOTAL_ARMS:
            note(f"COMPLETE: {arms}/{TOTAL_ARMS} arms finished; watchdog exiting")
            return

        cmds = python_command_lines()
        driver_alive = any("ll_16h.py" in c for c in cmds)
        worker_alive = any("noise_aware_sweep" in c for c in cmds)
        age = newest_csv_age_min()

        if not driver_alive and not worker_alive:
            note(f"DRIVER GONE ({arms}/{TOTAL_ARMS} arms done) -> restarting")
            start_driver()
        elif worker_alive and age is not None and age > STALE_MIN:
            note(f"WORKER STALLED: newest CSV {age:.0f} min old -> killing and restarting")
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                            "Where-Object { $_.CommandLine -like '*noise_aware_sweep*' } | "
                            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                           capture_output=True, text=True)
            start_driver()
        else:
            age_txt = "no CSV yet" if age is None else f"newest CSV {age:.0f} min old"
            note(f"ok  arms={arms}/{TOTAL_ARMS}  driver={driver_alive}  worker={worker_alive}  {age_txt}")

        time.sleep(CHECK_EVERY_S)


if __name__ == "__main__":
    main()
