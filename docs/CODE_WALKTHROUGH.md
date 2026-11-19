# Code Walkthrough – Autonomous Driving Stack, Explained Step by Step

This document explains every part of the project from zero. No prior knowledge of robotics, control theory or
Kalman filters is needed. Each section says **what** a module does in plain words, **where** it lives in the code,
and **why** it is built that way. Formulas are included, but always explained in words first.

---

## 0. The big picture: how a human drives, and how the code does it

When you drive a car you constantly answer four questions:

| Question | Human | This project | Folder |
|---|---|---|---|
| *Where am I?* | look around, feel the car | **Localization**: combine noisy sensors into one good estimate | `adstack/localization` |
| *Where do I want to go, and how?* | choose lane, plan to overtake, plan a parking manoeuvre | **Planning**: produce a path and a target speed | `adstack/planning` |
| *How do I turn the wheel and press the pedals to follow that plan?* | steering and pedal corrections | **Control**: steering angle + acceleration | `adstack/control` |
| *How does the car react?* | physics | **Vehicle model**: equations of motion | `adstack/vehicle` |

Because we have no real car, a **simulator** (`adstack/sim`) plays the role of reality: it moves a virtual car,
generates sensor readings, and records everything so we can measure how well the stack performs.

One cycle of the loop (repeated 20 times per second, i.e. every `dt = 0.05 s`):

```
true car state ──► sensors (noisy) ──► EKF estimate ──► controller ──► steering + acceleration ──► car moves
      ▲                                                                                              │
      └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

This loop lives in `adstack/sim/simulator.py → run_closed_loop()`.

---

## 1. Conventions you need everywhere

- **Pose** = position `(x, y)` + heading angle `yaw` (direction the car points, in radians; 0 = along the x-axis).
- The pose always refers to the **centre of the rear axle**. This is the standard reference point for car-like
  robots because the rear wheels do not steer, which makes the geometry simple.
- **`v`** = longitudinal speed. Negative `v` means reversing.
- **Steering angle `δ` (delta)** = angle of the front wheels. Positive = left.
- **Arc length `s`** = distance travelled along a path ("I am 123 m into the track").
- **Lateral error `e_y`** = how far the car is to the left (+) or right (−) of the path.
- `wrap_angle()` keeps angles between −π and π (so 359° and −1° are treated as the same direction).

---

## 2. The vehicle – `adstack/vehicle/models.py`

### 2.1 `VehicleParams` – the car's data sheet

A mid-size passenger car: wheelbase 2.85 m, length 4.70 m, width 1.85 m, mass 1700 kg, maximum steering angle 35°,
maximum steering speed 45°/s, maximum acceleration 3 m/s², maximum braking 7 m/s². Derived values such as
`min_turning_radius` are computed from these.

### 2.2 Two models of the same car

Think of two levels of realism:

**Kinematic bicycle model (`KinematicBicycle`)** – "tyres never slip". The two front wheels are merged into one,
the two rear wheels too (hence "bicycle"). If the wheels never slide sideways, the car moves on a circle whose radius
depends only on the steering angle:

$$\dot x = v\cos\psi,\quad \dot y = v\sin\psi,\quad \dot\psi = \frac{v\tan\delta}{L},\quad \dot v = a$$

In words: the car drives in the direction it points; it turns faster the more you steer and the faster you go.
Accurate at low speed (parking). Used inside the controllers because it is simple.

**Dynamic bicycle model (`DynamicBicycle`)** – "tyres behave like springs". At speed, a tyre only produces
sideways force if it slides a little (*slip angle* `α`). The model computes these forces
(`F_y = −C · α`, with cornering stiffness `C`) and then Newton's laws for sideways motion and rotation:

$$\dot v_y = \frac{F_{yf}\cos\delta + F_{yr}}{m} - r\,v_x,\qquad \dot r = \frac{l_f F_{yf}\cos\delta - l_r F_{yr}}{I_z}$$

(`r` = yaw rate, `I_z` = resistance against rotation, `l_f`/`l_r` = distance from the centre of gravity to the
front/rear axle.) Our car **understeers**: it turns slightly less than the kinematic model predicts – like a real car.

The dynamic model is used as the **"real" car** in fast scenarios. The controllers are designed on the simple model,
so they face a realistic mismatch, as in real development.

Detail: tyre slip is not defined at standstill (division by zero). Below 1–3 m/s, `derivatives()` therefore blends
smoothly into the kinematic behaviour (`v_blend`).

### 2.3 Actuators – the car does not obey instantly

`_PlantBase._update_actuators()` models what happens between the controller's command and the wheels:

- Steering follows the command with a small delay (time constant 0.08 s) and cannot turn faster than 45°/s.
- Acceleration/braking follows with a delay of 0.15 s.
- Everything is clipped to the physical limits.

This is why controllers cannot simply command "jump to 30° now".

### 2.4 Integration – moving forward in time

`step(accel_cmd, steer_cmd, dt)` advances the car by `dt` using the **Runge-Kutta 4** method (`_rk4`): it evaluates
the equations four times inside the step and averages them – much more accurate than simply multiplying by `dt`.
`observe()` returns the current `VehicleState` (rear-axle pose, speed, yaw rate, accelerations, actual steering).

`footprint()` returns the four corners of the car body – used for drawing and collision checks.

---

## 3. Localization – `adstack/localization/`

### 3.1 The problem

The controller needs to know where the car is. Sensors are imperfect:

| Sensor (`sensors.py`) | Measures | Weakness in the simulation |
|---|---|---|
| GNSS (GPS) | position, 10 times per second | ±0.8 m noise, can drop out (tunnel) |
| Gyroscope (IMU) | rotation speed (yaw rate) | noise + a constant **bias** of 0.02 rad/s (1.15°/s) |
| Accelerometer (IMU) | forward acceleration | noise |
| Wheel-speed sensor | speed | small noise |

`SensorSuite.read()` takes the true state from the simulator and returns these noisy readings.
`SensorConfig.gps_outages` defines time windows without GPS.

### 3.2 The idea of a Kalman filter (no math)

Imagine you walk with your eyes closed and count steps (prediction: smooth, but errors accumulate). Every few
seconds someone shouts your approximate position (measurement: no drift, but noisy). A Kalman filter combines both:

1. **Predict:** move the estimate forward using the motion sensors and grow the uncertainty a bit.
2. **Update:** when a measurement arrives, pull the estimate towards it – strongly if the measurement is precise
   compared to the current uncertainty, weakly if not.

It keeps not only an estimate `x` but also its uncertainty, the covariance matrix `P`.

"Extended" Kalman filter (EKF) means the motion equations are non-linear (sin, cos), so they are linearised
(Jacobian matrix `F`) at every step.

### 3.3 The EKF in this project – `ekf.py`

State (5 numbers): `[x, y, yaw, v, gyro_bias]`. The filter **estimates the gyro bias as well**. That is the trick
that keeps the heading accurate when GPS is gone.

`EKF.step(reading, dt)` does one cycle:

1. `predict(gyro, acc, dt)`
   - new position = old position + `v·cos(yaw)·dt`, `v·sin(yaw)·dt`
   - new heading = old heading + `(gyro − bias)·dt`
   - new speed = old speed + `acc·dt`
   - `P = F P Fᵀ + Q` (uncertainty grows; `Q` = how much we distrust the motion model)
2. `update_wheel_speed(v_meas)` – corrects the speed.
3. `update_gps(z)` – only if a GPS fix is available in this cycle; corrects the position (and indirectly heading and
   bias, because they are correlated with the position error).

Inside `_update()` (standard Kalman equations):

$$y = z - h(x),\quad S = HPH^\top + R,\quad K = PH^\top S^{-1},\quad x \leftarrow x + Ky$$

In words: `y` = surprise ("measurement minus expectation"), `S` = expected size of the surprise, `K` = how much of
the surprise we believe. The covariance update uses the *Joseph form*, which stays numerically stable.

**How we check it** (`sim/metrics.py → localization_metrics`): position error vs. raw GPS error, and the **NEES**
(normalized estimation error squared). If the filter is honest about its uncertainty, the mean position NEES is about
2. Too high = overconfident, too low = too pessimistic.

---

## 4. Planning – `adstack/planning/`

Planning answers "which path, at which speed?". There are three planners for three situations, plus shared tools.

### 4.1 `path.py` – `ReferencePath`, the common language

Every planner outputs a `ReferencePath`, and every controller consumes one.

- `from_waypoints(xs, ys)` draws a smooth **cubic spline** through a few points and samples it every 0.25–0.5 m.
  For every sample it stores `s` (distance along the path), `x`, `y`, `yaw` and `kappa` (**curvature** = 1/radius,
  positive for left turns).
- `from_poses(...)` builds a path from an already dense pose sequence (used for Hybrid A* output).
- `direction` = +1 forward, −1 reverse. `kappa` is stored "as seen by the car", so the steering angle that would keep
  the car on the path is always `atan(L · kappa)`.
- `project(x, y, s_hint)` finds the closest point on the path → returns `s` and the lateral error `e_y`. The search
  window around the last known position (`s_hint`) prevents jumping to a different part of a looping track.
- `to_cartesian(s, d)` converts "distance along the path + offset to the left" into `(x, y)` (used by the Frenet
  planner).
- `set_speed_profile(...)` computes how fast the car may go at every point:
  1. **Corner limit:** in a curve the sideways acceleration is `v²·κ`. Limit it to `a_lat_max` →
     `v ≤ sqrt(a_lat_max / |κ|)`.
  2. **Forward pass:** the car cannot accelerate faster than `a_acc` → `v_next ≤ sqrt(v² + 2·a_acc·Δs)`.
  3. **Backward pass:** the car must brake *before* a corner → same idea with `a_dec`, going backwards.
- `steering_rate_limit(...)` slows the car down where the curvature jumps, because the steering wheel cannot turn
  infinitely fast (important for parking paths).

### 4.2 `grid.py` – the map for parking

- `OccupancyGrid`: the parking lot as a grid of 10 cm cells, each free or occupied. Obstacles are added as polygons
  (`add_rectangle`, `add_polygon`, `add_border`).
- `compute_distance()` creates a **distance map**: for every cell, the distance to the nearest obstacle
  (`scipy.ndimage.distance_transform_edt`).
- `FootprintChecker.collision_free(x, y, yaw)`: places points every 25 cm on the outline of the car body and looks up
  their distance to obstacles. If every point is further away than a safety margin → no collision. Fast, because
  each check is just a table lookup.
- `astar_grid(...)`: classic **A\*** on the grid (shortest path for a point that can move in 8 directions).
- `dijkstra_cost_to_goal(...)`: distance from every cell to the goal, going around obstacles. Used as a "smart
  estimate" (heuristic) by Hybrid A*.

### 4.3 `reeds_shepp.py` – the shortest way for a car that can reverse

A car cannot move sideways and has a minimum turning radius. Reeds and Shepp proved (1990) that the shortest path
between two poses for such a car always consists of at most five pieces, each being **L** (full left turn),
**R** (full right turn) or **S** (straight), driven forwards or backwards.

- `all_paths(start, goal, radius)` tries all formula families (CSC, CCC, CCCC, CCSC, CCSCC, combined with mirror and
  time-reversal symmetries) and returns every valid candidate, sorted by length.
- `sample_path(...)` turns a candidate into a list of poses every 10 cm, plus the driving direction at each pose.

These paths ignore obstacles. Hybrid A* uses them for the final approach and checks them for collisions.
The tests verify that every generated path really ends at the goal (300 random cases).

### 4.4 `hybrid_astar.py` – parking manoeuvres

Normal A* plans for a point in a grid. A car is not a point and cannot turn on the spot. **Hybrid A\*** fixes this:

1. **Nodes are real car poses** `(x, y, yaw)`, not grid cells.
2. **Expansion = short drives:** from each pose, simulate 1.2 m of driving with 5 different steering angles,
   forwards and backwards (`_primitive`) → up to 10 children.
3. **Grid for bookkeeping:** poses are sorted into cells of 0.5 m × 0.5 m × 7.5° (`_key`) so the search does not
   revisit almost identical poses.
4. **Costs (`_step_cost`):** path length, ×1.5 for reversing, +5 for every change of direction, small penalties for
   steering and for changing the steering. This produces natural manoeuvres with few direction changes.
5. **Heuristic (`_heuristic`):** the Dijkstra distance from 4.2 – it already knows which way leads around the parked
   cars, so the search heads in the right direction.
6. **Analytic expansion (`_analytic`):** every 5 expansions (or when close to the goal), try the 6 shortest
   Reeds-Shepp paths from the current pose to the exact goal pose. If one is collision-free → done. This is what
   makes the car end exactly in the slot.
7. `_reconstruct` walks back through the parents and returns a `HybridAStarResult`.
8. `HybridAStarResult.segments()` cuts the path at each **cusp** (point where the car switches between forward and
   reverse) into separate `ReferencePath`s.

### 4.5 `frenet.py` – overtaking and avoiding obstacles on the road

**Frenet frame** = describe positions relative to the road instead of in x/y: `s` = how far along the road,
`d` = how far to the left of the right lane centre. Lane 1 centre is `d = 0`, lane 2 centre is `d = 3.5 m`.
In this frame "change lanes" simply means "change `d` from 0 to 3.5".

`FrenetPlanner.plan(state, obstacles, v_des)` runs 5 times per second:

1. **Start state** (`frenet_state`): where the car is in `(s, d)` and how its `d` is changing.
   *Important detail:* as long as the car tracks the previous plan closely (< 0.5 m), the new plan starts from the
   previous plan, not from the measured position. Otherwise planner and controller fight each other and the car
   overshoots during lane changes.
2. **Generate candidates:**
   - 7 target offsets `d_T` (lane centres plus small "nudges"),
   - 4 manoeuvre durations (2 to 5.5 s),
   - 7 target speeds (100 % … 0 % of the desired speed).
   The lateral shape is a **quintic polynomial** `d(s)` (`_quintic`): it starts with the current position, slope and
   curvature and ends at `d_T` with zero slope and curvature – a smooth, comfortable lane change.
3. **Collision check with prediction** (`_time_at`, `_check`): for each candidate we compute *when* the car would
   reach each point (with the candidate's speed) and where every obstacle will be at that moment (obstacles move with
   constant speed, `Obstacle.s_at`). Overlap of the two rectangles → candidate rejected. Too large sideways
   acceleration → rejected.
4. **Score the rest** (lower = better):
   - distance from the nearest lane centre (no driving on the lane marking),
   - keep-right rule (penalty for the left lane),
   - lateral jerk (comfort), manoeuvre duration,
   - deviation from the desired speed,
   - proximity to obstacles (keep a comfortable gap),
   - consistency with the previous plan (no zig-zagging between options).
5. **Choose** the cheapest free candidate. If nothing is free with the normal margins, try again with minimal
   margins; if still nothing → **emergency brake** in the current lane (`_stopping_profile`).
6. **Output** a `ReferencePath` with a speed profile → handed to the controller.

The `mode` string (`cruise`, `overtake / avoid`, `follow / slow down`, `emergency brake`) is only for display.

---

## 5. Control – `adstack/control/`

All controllers share the interface in `base.py`:

```python
out = controller.control(state, path)  # → ControlOutput(accel, steer, info)
```

### 5.1 `SpeedController` (in `base.py`) – the gas and brake pedal

Used by Pure Pursuit and Stanley (the MPC does its own speed control).

1. Look at the target speed slightly **ahead** (0.6 s preview), so braking starts before the corner.
2. **Feed-forward:** the acceleration the speed profile itself requires.
3. **PI feedback:** correct the remaining speed error (`kp = 1.2`, `ki = 0.3`).

### 5.2 Pure Pursuit – "aim at a point ahead" (`pure_pursuit.py`)

Like looking at a point a few metres ahead on the road and steering towards it.

1. Look-ahead distance grows with speed: `L_d = 0.15 s · v + 3 m`.
2. Find the path point at distance `L_d` from the rear axle.
3. Compute the circular arc through the rear axle and that point. Its steering angle is

$$\delta = \arctan\frac{2L\sin\alpha}{L_d}$$

(`α` = angle between the car's heading and the line to the target point.)

Simple and robust, but it cuts corners slightly (the look-ahead point is "inside" the curve), and too short a
look-ahead makes it oscillate at speed.

### 5.3 Stanley – "fix heading and offset at the front axle" (`stanley.py`)

Used by Stanford's car that won the 2005 DARPA Grand Challenge.

$$\delta = \underbrace{\theta_e}_{\text{heading error}} - \underbrace{\arctan\frac{k\,e_{fa}}{k_{soft}+v}}_{\text{pull back to the path}} + \underbrace{\arctan(L\kappa)}_{\text{feed-forward}} + \underbrace{k_r\,(v\kappa - r)}_{\text{yaw damping}}$$

1. Point the wheels parallel to the path (heading error `θ_e`).
2. Steer towards the path, more strongly the further away the front axle is (`e_fa`), less strongly at high speed.
3. Feed-forward: the steering a perfect car would need for this curvature.
4. Yaw-rate damping: compares the actual rotation speed with the expected one – this is what makes Stanley robust
   when the car understeers more than expected.

### 5.4 MPC – "plan the next 2 seconds, execute the first step, repeat" (`mpc.py`)

**Model Predictive Control** is like a chess player who thinks a few moves ahead:

1. **Predict:** with the kinematic model, how would the car move over the next 20 steps × 0.1 s = 2 s for any
   sequence of steering and acceleration commands?
2. **Optimise:** find the command sequence that keeps the predicted car closest to the reference (position, heading,
   speed), with smooth steering, while respecting the limits (max. steering, max. steering speed, max.
   acceleration/braking).
3. **Apply only the first command**, then repeat the whole thing in the next cycle with fresh measurements.

How the code does it:

- `_reference()` builds the 21 reference points the car *should* reach, by walking along the path with the speed
  profile.
- `_linearise()`: the car model contains sin/cos, which makes optimisation hard. Around the previously predicted
  trajectory the model is approximated by a straight-line (linear) version `x_{k+1} = A_k x_k + B_k u_k + c_k`.
  Because this is redone every cycle with different `A_k`, it is called **LTV-MPC** (linear time-varying).
- The result is a **quadratic program (QP)**: minimise a sum of squared errors subject to linear constraints:

$$\min \sum_k \|x_k - x_k^{ref}\|_Q^2 + \|u_k - u_k^{ref}\|_R^2 + \|u_k - u_{k-1}\|_{R_d}^2$$

  `Q` = how much each error matters, `R` = how much using the actuators costs, `R_d` = how much sudden changes cost.
- `_build_static()` builds the parts of the QP that never change, once. Every cycle only the numbers are updated and
  the OSQP solver is **warm-started** from the last solution → about 3 ms per solve.
- If the solver ever fails, the shifted previous plan is used as a fallback.
- Reversing works automatically: in reverse segments the reference speed is negative.

---

## 6. The simulator – `adstack/sim/`

### 6.1 `run_closed_loop()` in `simulator.py`

For every time step:

1. **Sense & localise:** either use the true state, or (with `use_ekf=True`) generate noisy sensor readings and run the
   EKF.
2. **Control:** `controller.control(estimate, path)`.
3. **Log:** true position, errors (always measured against the *true* state), commands, solve times …
4. **Check termination:** lap finished (closed track) or segment end reached (open path). For parking, the next
   segment starts; with `steer_at_standstill=True` the wheels are first turned to the new segment's curvature while
   the car stands still – like a parking assistant does at a cusp.
5. **Move the car:** `vehicle.step(accel, steer, dt)`.

Everything is stored in a `SimLog`; `log.arrays()` turns it into NumPy arrays.

### 6.2 `scenarios.py` – the four test drives

| Function | Scenario |
|---|---|
| `racetrack()`, `run_racetrack()` | 790 m closed track, up to 90 km/h, dynamic car model |
| `perpendicular_parking()` | Parking lot with two rows of cars; reverse into one free 2.7 m slot |
| `parallel_parking()` | Street with parallel parked cars and a 7.5 m gap at the kerb |
| `plan_parking()`, `run_parking()` | Hybrid A* plan → split into segments → MPC tracking with the kinematic model |
| `two_lane_road()`, `default_obstacles()`, `run_avoidance()` | 620 m two-lane road; slow truck, cones, a car in the left lane, a broken-down car; Frenet planner re-plans every 4 control cycles (5 Hz) |

### 6.3 `metrics.py` – turning logs into numbers

- `tracking_metrics()`: RMS and maximum lateral error, heading error, speed error, lap time, maximum lateral
  acceleration (comfort/safety), steering rate (smoothness), solver time (real-time capability).
- `localization_metrics()`: GPS vs. EKF error, heading error, bias estimate, NEES, maximum error during the outage.

---

## 7. Visualisation – `adstack/viz/plots.py`

- `CarArtist` draws the car body and four wheels; the front wheels turn with the steering angle.
- `plot_*` functions create the static figures in `assets/`, `animate_*` functions create the GIFs.
- The avoidance animation shows all candidate paths of the current plan: grey = free, red = would collide,
  green = chosen.

---

## 8. Putting it all together – `scripts/run_all.py`

```bash
python scripts/run_all.py              # all scenarios, figures and GIFs (~1.5 min)
python scripts/run_all.py --no-gif --only parking
```

| Function | Does |
|---|---|
| `racetrack()` | Runs all three controllers under three conditions (nominal, EKF in the loop, heavier car with less grip) → `results/racetrack.md` |
| `localization()` | One lap with a 10 s GPS outage → `results/localization.md`, `assets/localization.png` |
| `parking()` | Both parking scenarios, collision check of the driven path → `results/parking.md` |
| `avoidance()` | Frenet planner with all three controllers → `results/avoidance.md` |

---

## 9. Tests – `tests/`

The tests check **physics and behaviour**, not just "does it run":

| Test file | Examples of what is checked |
|---|---|
| `test_vehicle_and_path.py` | Kinematic car drives exactly the theoretical circle radius `L / tan(δ)`; dynamic car matches the yaw rate of linear single-track theory; steering rate limit is respected; spline circle has the correct length and curvature; speed profile never exceeds the lateral/longitudinal limits; reverse-driving curvature convention |
| `test_planning.py` | Every Reeds-Shepp path ends exactly at the goal; A* goes around a wall; Hybrid A* parking path is collision-free and contains a direction change; Frenet planner keeps the lane when free, changes lane around an obstacle, slows down when both lanes are blocked |
| `test_control_and_localization.py` | All three controllers finish a lap with RMS error < 0.3 m; MPC respects its limits and solves fast enough for 20 Hz; EKF beats raw GPS and finds the gyro bias; parking ends within 15 cm of the goal without collision; avoidance scenario without collision |

```bash
pytest          # 20 tests, ~20 s
```

---

## 10. Reading the results

- **RMS lateral error** – typical distance from the path ("how precise").
- **Max lateral error** – worst moment ("how safe").
- **RMS steer rate** – how nervous the steering is ("how comfortable").
- **Solve time** – must stay well below the 50 ms cycle time to run in real time.
- **Min. clearance** – smallest gap to any obstacle in the avoidance scenario.

Main lessons from the benchmark (numbers in the README and `results/`):

1. With a perfect state estimate, the **MPC** is the most precise and also follows the speed profile best.
2. When the real car differs from the model (heavier, less grip), **Stanley** with yaw-rate feedback is more robust:
   the MPC relies on its (kinematic) prediction model.
3. With the EKF in the loop, all controllers steer more nervously because the estimate jumps slightly with every GPS
   fix.

---

## 11. Glossary

| Term | Plain meaning |
|---|---|
| Pose | Position + heading |
| Curvature κ | 1 / turning radius; 0 = straight |
| Kinematic vs. dynamic model | "Tyres never slip" vs. "tyres produce force by slipping a little" |
| Understeer | The car turns less than the steering angle suggests |
| EKF | Filter that fuses a motion prediction with noisy measurements, including uncertainty |
| Covariance P | The filter's own estimate of how uncertain it is |
| Bias | A constant offset of a sensor (here: the gyroscope) |
| NEES | Check whether the filter's uncertainty matches its real error |
| A* / Hybrid A* | Graph search for shortest paths / version that respects car kinematics |
| Heuristic | Optimistic guess of the remaining cost; guides the search |
| Reeds-Shepp path | Shortest path for a car that can drive forwards and backwards |
| Cusp | Point where the car switches between forward and reverse |
| Frenet frame | Coordinates along the road (`s`) and across it (`d`) |
| Quintic polynomial | 5th-order polynomial; allows fixing position, slope and curvature at both ends |
| Feed-forward | Steering/acceleration computed from the plan itself, before any error occurs |
| Feedback | Correction based on the measured error |
| MPC | Repeatedly optimise the next few seconds and apply only the first command |
| QP | Quadratic program: minimise a quadratic cost under linear constraints |
| Warm start | Start the solver from the last solution to make it faster |
| RMS | Root mean square – a typical size of an error over time |

