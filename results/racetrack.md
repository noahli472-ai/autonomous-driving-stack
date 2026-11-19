# Race track benchmark

| Condition | Controller | RMS lateral error [m] | Max lateral error [m] | RMS heading error [deg] | RMS speed error [m/s] | Lap time [s] | Max lat. accel [m/s²] | RMS steer rate [deg/s] | Mean / max solve [ms] |
|---|---|---|---|---|---|---|---|---|---|
| nominal | Pure Pursuit | 0.202 | 0.410 | 1.57 | 2.55 | 52.5 | 5.84 | 2.75 | < 0.1 |
| nominal | Stanley | 0.157 | 0.409 | 1.62 | 2.55 | 52.1 | 5.50 | 3.04 | < 0.1 |
| nominal | MPC | 0.129 | 0.339 | 1.72 | 2.43 | 51.3 | 7.20 | 4.29 | 3.0 / 10.2 |
| EKF in the loop | Pure Pursuit | 0.464 | 1.213 | 3.79 | 2.64 | 52.8 | 11.66 | 21.53 | < 0.1 |
| EKF in the loop | Stanley | 0.370 | 0.967 | 3.22 | 2.63 | 52.5 | 10.50 | 25.86 | < 0.1 |
| EKF in the loop | MPC | 0.435 | 1.147 | 4.69 | 2.51 | 51.8 | 14.13 | 28.91 | 2.8 / 6.2 |
| +25% mass, -30% tyre stiffness | Pure Pursuit | 0.372 | 0.966 | 3.07 | 2.54 | 52.9 | 8.25 | 4.89 | < 0.1 |
| +25% mass, -30% tyre stiffness | Stanley | 0.192 | 0.621 | 2.75 | 2.55 | 52.5 | 5.54 | 3.43 | < 0.1 |
| +25% mass, -30% tyre stiffness | MPC | 0.290 | 0.682 | 3.22 | 2.43 | 51.5 | 8.32 | 7.08 | 3.0 / 5.6 |
