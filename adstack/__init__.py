"""adstack - a compact autonomous driving stack in Python.

Modules
-------
vehicle       kinematic / dynamic single-track models with actuator dynamics
localization  sensor simulation (GNSS, IMU, wheel speed) and an EKF
planning      reference paths, grid A*, Hybrid A* + Reeds-Shepp, Frenet planner
control       Pure Pursuit, Stanley, LTV-MPC (OSQP)
sim           closed-loop simulator, scenarios and metrics
viz           figures and animations
"""

__version__ = "1.0.0"
