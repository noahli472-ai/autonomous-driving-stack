"""Minimal example: one lap on the race track with the MPC and the EKF in the loop.

python examples/quickstart.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adstack.control import MPC  # noqa: E402
from adstack.sim import scenarios  # noqa: E402
from adstack.sim.metrics import tracking_metrics  # noqa: E402


def main() -> None:
    track = scenarios.racetrack(v_max=25.0)
    log = scenarios.run_racetrack(MPC(), track, use_ekf=True)
    for key, value in tracking_metrics(log, track).items():
        print(f"{key:28s} {value:.3f}" if isinstance(value, float) else f"{key:28s} {value}")


if __name__ == "__main__":
    main()
