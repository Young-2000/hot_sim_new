<h1 align="center">🌡️ Thermal Studio</h1>

<p align="center">
  <strong>Local-first 3D thermal simulation workbench for engineering analysis.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="#quick-start-windows"><img alt="Windows, Linux, and macOS" src="https://img.shields.io/badge/Platforms-Windows%20%7C%20Linux%20%7C%20macOS-555555"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img alt="CPU or CUDA" src="https://img.shields.io/badge/Compute-CPU%20%7C%20CUDA-76B900?logo=nvidia&logoColor=white"></a>
</p>

<p align="center">
  <a href="README.md"><strong>English</strong></a> ·
  <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <a href="#capabilities">Capabilities</a> ·
  <a href="#quick-start-windows">Quick start</a> ·
  <a href="#simulation-assistant">Simulation assistant</a> ·
  <a href="#codex-plugin">Codex plugin</a> ·
  <a href="#development-and-validation">Development &amp; validation</a>
</p>

<hr>

Import STL / STEP models on your local machine, configure materials, heat sources, and cooling conditions, then compute and inspect 3D temperature changes. The project includes a Python simulation service, a browser UI, and Codex plugin source for starting the service.

Core calculations use Gmsh tetrahedral meshing, scikit-fem finite-element assembly, and SciPy / CuPy solvers. Thermal Studio is suitable for solid-conduction cases and interactive demonstrations; see “Physics model and known limitations” below for the scope of advanced physical features.

## Capabilities

| Area | Current scope |
| --- | --- |
| Models and materials | STL, STEP / STP import, geometry quality checks, material presets, custom properties, box-based material regions, and component materials |
| Heat sources and boundaries | Surface and point heat sources, embedded placement, brush selection and directional face selection, multiple sources, start/stop times, power curves, temperature control, convection, and radiation |
| 3D solving | Transient and steady-state solid conduction, CPU / CUDA solving, phase-change effective heat capacity, effective contact resistance, and air-gap conduction |
| Results | Temperature animation, maximum/minimum/average temperature, axial slicing, thermal-expansion displacement estimates, CSV, Markdown / PDF reports, and ParaView data |
| Case management | Save and load configurations, run history, progress reporting, and cancellation |
| Simulation assistant | A local rule parser, or Codex CLI / Responses API generation of reviewable configuration drafts |
| Supporting analysis | CPU / motherboard / heatsink thermal networks, heat-transfer coefficient fitting, safe-power estimates, and heat-pump / refrigeration-cycle COP estimates |

## Quick start (Windows)

Prepare Python 3.11 or later and a WebGL-capable browser. From the repository root, run:

```powershell
git clone https://github.com/tianyuzong/hot_sim_new.git
cd hot_sim_new
powershell -NoProfile -ExecutionPolicy Bypass -File .\Install.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start.ps1 -Device auto
```

`Install.ps1` creates `.venv` and installs the dependencies in [requirements.txt](requirements.txt). The first installation requires network access and includes CUDA-related packages. `-Device auto` falls back to CPU when a GPU is unavailable; CPU operation does not require an NVIDIA GPU.

When the service starts, open the [local simulation UI](http://127.0.0.1:8765/). The service listens on the local machine only. Three.js and other page dependencies are vendored in `static/vendor/`, so ordinary local simulation does not depend on a CDN or Codex.

### Starting, stopping, and compute devices

```powershell
# Force CPU mode
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start.ps1 -Device cpu

# Require an available NVIDIA CUDA environment; fail if detection does not succeed
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start.ps1 -Device cuda

# Use another port and do not open a browser automatically
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start.ps1 -Device auto -Port 8766 -NoBrowser

# Check the actual compute device and status on the default port
Invoke-RestMethod http://127.0.0.1:8765/api/health

# Stop the service and any active computation; use the matching -Port for custom ports
powershell -NoProfile -ExecutionPolicy Bypass -File .\Stop.ps1 -Port 8765
```

| Entry point | Default device behavior |
| --- | --- |
| `Start.bat` or the root `Start.ps1` without `-Device` | `cuda`; startup fails if CUDA is unavailable |
| `plugins/thermal-studio/scripts/start.ps1` | `auto`; falls back to CPU if CUDA is unavailable |
| Direct Python service startup | Reads `THERMAL_DEVICE`; defaults to `auto` when unset |

The startup scripts override `THERMAL_DEVICE` with the `-Device` value, so pass the device explicitly when switching modes through a script. CUDA accelerates sparse linear solving; model processing, mesh generation, finite-element assembly, and export still run on the CPU.

Closing the browser does not stop the service. After changing the device or environment variables, stop the existing service and restart it from a shell with the new variables. Logs are stored in `.runtime/server.log` and `.runtime/server-error.log`.

## First 3D case

1. **Import a model.** Upload a closed STL or STEP / STP file. STL files require a coordinate unit; STEP units are interpreted from the file declaration. An additional scale factor changes the final size. Check the actual dimensions after import.
2. **Set materials.** Choose a base material. For multiple materials, add box regions. Preset properties are examples and should be adjusted for the material grade and temperature range.
3. **Add a heat source.** Surface heat sources require brushed or direction-selected heated faces; point heat sources require a valid position. Enter total power and start/end times. Setting only a material and duration does not create a heat source.
4. **Set cooling.** Configure the ambient temperature and heat-transfer coefficient `h`, or create designated cooling regions. `h=0` disables that convection term; whether a heat-source face also receives default cooling is controlled by the corresponding option.
5. **Configure the run.** Set the initial temperature, duration, time step, save interval, and mesh size, then run the simulation.
6. **Review the results.** Inspect the temperature animation, slices, material volumes, heat-source mapping, and energy checks, then export or save the case as needed.

UI fields use their displayed units. API coordinates, radii, and mesh sizes with `_m` use meters; power uses W, time uses s, and temperature uses °C. The repository does not include models from the local `data/` directory; example dimensions and mass depend on the actual import settings.

Local page entry points: [3D simulation](http://127.0.0.1:8765/) · [CPU thermal workbench](http://127.0.0.1:8765/cpu.html) · [UI guide](http://127.0.0.1:8765/guide.html). These links require the service to be running; update the address when using another port.

## Simulation assistant

Load a model first, then open the “Simulation assistant”. Generation creates a draft only. Choose “Apply and edit” to return to the main UI and inspect parameters and selections; choose “Confirm and simulate” to submit the run.

### Parser modes and connection methods

| Page mode / configuration | Behavior |
| --- | --- |
| Local rule parser | No network access required; recognizes parameters using built-in rules. Review generated content for complex requests. |
| Codex, `THERMAL_CODEX_PROVIDER=cli` (default) | Calls the official local `codex exec` and reuses the CLI login state; reports a clear error if the CLI is missing. |
| Codex, `THERMAL_CODEX_PROVIDER=api` | Uses the OpenAI Responses API and requires `OPENAI_API_KEY`. |
| Codex, `THERMAL_CODEX_PROVIDER=auto` | Tries the CLI first and uses the API only when the CLI is missing; a failed CLI request does not automatically switch to the API. |

CLI mode requires the official Codex CLI to be installed and logged in. Check it in a terminal with `codex --version` and `codex login status`. The program also searches the official CLI installation directories on Windows; use `THERMAL_CODEX_COMMAND` to provide an executable path. Calls use a temporary session and a read-only sandbox.

When Codex is used, the natural-language target, current configuration, and model geometry summary are sent to the selected service. The 3D simulation still runs locally. A Codex connection failure is shown as an error and does not silently fall back to the local rule parser.

### Example prompt with a heat source

After importing a model, select **Codex** and enter:

```text
Create a transient heat-conduction case for the currently loaded model and keep the current mesh size.
Use copper as the base material, with an initial temperature of 25°C and an ambient temperature of 25°C.
Keep exactly one surface heat source named "Top heater": total power 20 W,
active from 0 to 600 seconds, with the model's top (+Z) exterior faces selected as the heated surface.
Apply convection to all exterior faces with h=10 W/(m²·K), including the heat-source face.
Set the total simulation time to 600 seconds, the time step to 5 seconds, and save one frame every 5 seconds.
Do not use a power curve, temperature control, radiation, phase change, or additional contact resistance.
If no suitable top heated face exists, ask me to select it again instead of generating a configuration without a heat source.
```

The draft should contain a **20 W heat source, a 0–600 s active interval, and a non-zero heated-face count**. Before confirming, use “Apply and edit” to check the selection, especially the model orientation and dimensions.

“Top / bottom, left / right, front / back, and all exterior faces” are converted by local geometry code into actual face IDs. Directional selection uses exterior triangle faces within the outermost 3% along that direction whose normals point toward it; it does not mean an arbitrary complete upper half of a shape. A missing heat source, empty selection, invalid face ID, or model mismatch prevents the draft from being confirmed.

### Environment variables

Set these before starting the service. Restart the service after changing them.

| Variable | Purpose and default |
| --- | --- |
| `THERMAL_CODEX_PROVIDER` | `cli` / `api` / `auto`, default `cli` |
| `THERMAL_CODEX_COMMAND` | Optional path to the Codex CLI executable |
| `THERMAL_CODEX_MODEL` | Optional CLI model; uses the existing CLI configuration when unset |
| `THERMAL_CODEX_TIMEOUT_S` | CLI wait limit, default `180` seconds, allowed range `15–600`; does not control API timeouts |
| `OPENAI_API_KEY` | Required for API mode; do not write it to the repository |
| `OPENAI_MODEL` | API model; the code default is `gpt-5.2` |
| `OPENAI_BASE_URL` | API service root; default `https://api.openai.com/v1` |
| `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` | Explicit proxy settings; the CLI child process prefers to inherit them |
| `THERMAL_DEVICE` | `auto` / `cpu` / `cuda` for direct Python startup; startup scripts override it |
| `THERMAL_CUDA_DEVICE` | CUDA device index, default `0` |
| `THERMAL_STUDIO_ROOT` | Full application root when the plugin is placed outside the repository |

For example, increase the CLI wait limit and start in CPU mode:

```powershell
$env:THERMAL_CODEX_PROVIDER = 'cli'
$env:THERMAL_CODEX_TIMEOUT_S = '300'
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start.ps1 -Device cpu
```

When proxy variables are not explicitly configured, the CLI child process reads the enabled Windows manual proxy and adds local addresses to the direct-connection list; it does not change the system or other processes' proxy settings. Environments configured only with a PAC script need to provide a usable proxy environment separately. The UI displays the wait duration and distinguishes startup failures, timeouts, and configuration-validation failures.

## Codex plugin

The plugin source is in [plugins/thermal-studio](plugins/thermal-studio/README.md). It contains `.codex-plugin/plugin.json`, the skill description, and startup / health-check scripts. It calls the service in this repository and does not package the Python environment, model cache, or computation results; cloning the repository and running `Install.ps1` do not install the plugin into Codex automatically.

After installing the root dependencies, run the plugin scripts from the **repository root**:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\plugins\thermal-studio\scripts\start.ps1 -Device auto
powershell -NoProfile -ExecutionPolicy Bypass -File .\plugins\thermal-studio\scripts\health-check.ps1
```

The plugin startup script supports `-Port`, `-Device`, and `-NoBrowser`. For placement outside the repository, see the [plugin README](plugins/thermal-studio/README.md).

## Save, export, and backup

| Directory / file | Contents |
| --- | --- |
| `data/models/` | Uploaded models, import metadata, and surface geometry |
| `data/projects/` | Saved case parameters and selection snapshots |
| `data/jobs/` | Each computation's inputs, logs, mesh, temperature fields, and exported results |
| `.runtime/` | Service logs, process information, and temporary runtime files |
| `.venv/` | The local Python virtual environment |

These directories are excluded by [.gitignore](.gitignore). **Pushing the source to GitHub does not back up models, saved cases, or computation results**; back up `data/` separately when migrating, then reinstall dependencies on the new machine.

After extracting a complete 3D result ZIP, open `temperature.xdmf` in ParaView and keep `thermal-fields.h5` in the same directory. Color by `Temperature_C`. Coordinates are in m, time is in s, and temperature is in °C. `Material_ID` maps to the material numbers in the result; `audit.json` records material volumes, heat-source mapping, compute device, and numerical checks.

## Physics model and known limitations

The 3D transient solver uses P1 tetrahedral finite elements, a consistent mass matrix, and backward-Euler time integration. Convection boundaries take a prescribed heat-transfer coefficient; the program does not solve wind speed, flow fields, or fluid pressure. STEP box material regions participate in CAD partitioning; STL box regions are assigned by element centers, so boundary accuracy depends on the mesh.

The following limitations still apply:

| Feature | Current limitation |
| --- | --- |
| Multiple independent STEP entities | Component-ID mapping has a known issue and may place multiple entities in the same component, affecting component materials and air-gap conduction; material assignment cannot be considered correct based on the UI alone |
| Power curves and phase change | Power-curve breakpoints are not automatically added to the time grid, so crossing a breakpoint can introduce input-energy error. Effective heat capacity is updated from the previous temperature, and crossing a phase-change range can fail the energy check |
| Contact resistance and air gaps | Contact resistance is approximated by modifying material conductivity and is mesh-dependent. Air gaps use simplified geometric pairing and `k·A / gap`, without complete occlusion, orientation, or flow handling |
| Thermal expansion displacement | A free-expansion estimate relative to the center; it does not include mechanical constraints, stress, or a full thermoelastic solve |
| Steady-state result audit | Some steady-state report energy fields still use transient terminology, and units and convergence checks need refinement; do not use them directly as a transient energy audit |
| CPU air- and water-cooling workbench | A lumped-parameter thermal network; current water-flow unit conversion and explicit time-integration stability have known issues, and large time steps can produce abnormal temperatures. Quantitative results require correction and review |
| Heat-transfer fitting and cycle estimates | Based on lumped-parameter or ideal-temperature-difference models; they do not include complete working-fluid properties or equipment performance curves |

For initial validation, use a single material, constant power, and a convection boundary. The program does not automatically perform mesh or time-step convergence analysis; compare different discretizations and check actual dimensions, material volumes, total input power, and key temperatures. A small energy residual alone does not prove that the physical model or local temperatures are correct.

## Development and validation

From the repository root:

```powershell
# Install development test dependencies; not required to start the service
.\.venv\Scripts\python.exe -m pip install pytest httpx

# Python regression tests; CUDA is not required
$env:THERMAL_DEVICE = 'cpu'
.\.venv\Scripts\python.exe -m pytest tests -q

# Frontend tests require Node.js on the local machine
node --test tests/test_agent_ui.cjs
node --check static/app.js
```

Tests cover basic physical cases, geometry quality, extended configurations, CPU thermal networks, Agent connections, heat-source selection, and frontend draft validation. They do not mean that all advanced models above have completed engineering validation.

There is also an optional API integration script. Start the service on the default `8765` port first, then run:

```powershell
.\.venv\Scripts\python.exe tests/integration_api.py
```

The script creates a test model and case, submits a real computation, checks exports, and rewrites `validation-api.json`. [VALIDATION.md](VALIDATION.md), `validation-api.json`, and `validation-wukong.json` are historical validation records; their models and parameters do not represent the current local import state.

### Source map

| Path | Responsibility |
| --- | --- |
| [server.py](server.py) | Local API, model uploads, case storage, task management, and download endpoints |
| [geometry.py](geometry.py) | Geometry processing, quality checks, CAD partitioning, and volume meshing |
| [solver.py](solver.py) | Finite-element assembly, time integration, boundary conditions, slicing, displacement estimates, and export |
| [analysis.py](analysis.py) | CPU thermal networks, heat-transfer fitting, power estimates, and cycle estimates |
| [agent.py](agent.py) | Natural-language parsing, CLI / API connections, face-selection mapping, and configuration validation |
| [schemas.py](schemas.py) | Parameter models, validation, and material presets |
| [runtime.py](runtime.py), [worker.py](worker.py) | Runtime directories, worker processes, and computation tasks |
| [static/](static/) | 3D UI, CPU workbench, user guide, and local frontend dependencies |
| [tests/](tests/) | Python, frontend, and API integration validation |
| [plugins/thermal-studio/](plugins/thermal-studio/README.md) | Codex plugin source and helper scripts |

## FAQ

| Symptom | What to do |
| --- | --- |
| Double-clicking `Start.bat` reports CUDA / CuPy / DLL errors | From the repository root, use `Start.ps1 -Device auto` or `-Device cpu`. For GPU mode, check the matching CUDA environment |
| Startup reports that the `solver` module is missing | Switch to the repository root before running the startup script |
| The page does not open or the port is occupied | Check `.runtime/server-error.log`, query `/api/health`, or start with `-Port 8766` |
| Codex cannot find the CLI or connect | Check the CLI installation, login state, and proxy. API users must explicitly set the provider and key |
| Codex requests time out | Check the connection and proxy first, then adjust `THERMAL_CODEX_TIMEOUT_S` if needed; increasing the timeout does not fix a failed connection |
| The UI says “Please add at least one heat source” | Check that the draft really created a heat source and that a surface source has selected faces or a point source has a position; use the complete example above |
| Model size, mass, or temperature rise looks wrong | Check the STL unit, STEP file unit, additional scale factor, material properties, region volumes, and heat-source selection |

Dependency versions are listed in [requirements.txt](requirements.txt), and third-party component notices are in [THIRD_PARTY.md](THIRD_PARTY.md).
