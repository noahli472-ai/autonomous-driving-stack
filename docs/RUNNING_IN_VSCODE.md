# Running the Autonomous Driving Stack in VS Code (Windows)

A cheat sheet: one-time setup, the commands to run, where the results end up, and fixes for the
usual errors.

---

## Quick start (after the one-time setup)

1. Open VS Code → *File → Open Folder* → the `autonomous-driving-stack` folder
   (the one that contains `README.md` and the `adstack` folder).
2. *Terminal → New Terminal*. The prompt must start with `(.venv)`.
   If not: `.venv\Scripts\Activate.ps1`
3. Run:

```powershell
python examples/quickstart.py
```

After a few seconds the metrics of one lap (MPC + EKF) are printed, e.g. `finished True`.

---

## All commands

| Command | What it does | Duration |
|---|---|---|
| `python examples/quickstart.py` | One lap on the race track with MPC and EKF, metrics in the terminal | a few s |
| `python scripts/run_all.py --no-gif` | All four scenarios: race track, localisation, parking, avoidance | ~40 s |
| `python scripts/run_all.py` | Same, plus all GIF animations | ~1.5 min |
| `python scripts/run_all.py --only parking` | Only selected scenarios: `racetrack`, `localization`, `parking`, `avoidance` (several allowed) | varies |
| `python scripts/run_all.py --only avoidance --no-gif` | Options can be combined | varies |
| `pytest` | Runs the 20 tests | ~20 s |
| `ruff check .` | Linting (needs `pip install ruff`) | < 1 s |

The ▶ button in VS Code works for `examples/quickstart.py` and `scripts/run_all.py`,
**not** for files inside `adstack/` (they are parts of a package).

---

## Where are the results?

The scripts **do not open windows**; everything is saved to files:

| Folder | Content | How to view in VS Code |
|---|---|---|
| `results/` | Tables with the metrics (`racetrack.md`, `localization.md`, `parking.md`, `avoidance.md`) and `racetrack.json` | Right-click a `.md` file → *Open Preview* |
| `assets/` | Figures (`.png`) and animations (`.gif`) | Click the file |

Running `run_all.py` overwrites the files in `results/` and `assets/`. They are part of the
repository; `git status` shows them as changed afterwards. To discard the changes:
`git checkout -- results assets`.

---

## One-time setup

### 1. Allow PowerShell scripts (once per Windows user)

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Confirm with `J` (German Windows) or `Y` (English Windows).

### 2. Get the code

```powershell
cd "$HOME\Desktop\Coding Projects"
git clone https://github.com/noahli472-ai/autonomous-driving-stack.git
cd autonomous-driving-stack
```

Without git: on GitHub *Code → Download ZIP*, unzip, and `cd` into the folder that directly
contains `README.md` (ZIP downloads often create a nested folder of the same name).

### 3. Create the virtual environment and install

Run these **inside the project folder**:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

Check: `.venv\Scripts\python.exe examples\quickstart.py` prints the metrics.

### 4. Connect VS Code to the environment

1. *File → New Window*, then *File → Open Folder* → the project folder
   (use a separate window from the pendulum project).
2. Install the **Python** extension (Microsoft) if it is missing.
3. `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the entry with `.venv` of this folder
   (`.\.venv\Scripts\python.exe`). If it is not listed: *Enter interpreter path…* →
   `.venv\Scripts\python.exe`.
4. Close old terminals (trash icon) and open a new one. It should now start with `(.venv)`.

---

## Troubleshooting

| Error | Cause → fix |
|---|---|
| `[WinError 5] Access denied: 'C:\WINDOWS\System32\.venv'` | The terminal is in the Windows system folder. Open the project folder in VS Code (step 4) or `cd` into it. Do not run VS Code as administrator. |
| `...Activate.ps1 cannot be loaded because running scripts is disabled` | Run step 1 (`Set-ExecutionPolicy ...`). |
| `ModuleNotFoundError: No module named 'numpy'` / `'osqp'` / `'scipy'` | Packages are not installed in the Python that runs. Run step 3 in the project folder and select the `.venv` interpreter (step 4). |
| `ModuleNotFoundError: No module named 'adstack'` | Wrong folder (often the outer folder of a nested ZIP) or `pip install -e .` missing. |
| `attempted relative import with no known parent package` | A file inside `adstack/` was started with ▶. Use the commands above. |
| "Nothing happens" after `run_all.py` | Intended: look into `results/` and `assets/`. |
| `.venv` created in the wrong place | Delete it (`Remove-Item -Recurse -Force .venv` in that folder) and repeat step 3 inside the project folder. |

Which Python is running?

```powershell
python -c "import sys; print(sys.executable)"
```

It must end with `...\autonomous-driving-stack\.venv\Scripts\python.exe`.

---

## macOS / Linux

Same steps with these differences: `python3 -m venv .venv`, activate with
`source .venv/bin/activate`, and the interpreter path is `.venv/bin/python`.
