<h1 align="center">🌡️ Thermal Studio</h1>

<p align="center">
  <strong>面向本地工程分析的三维热仿真工作台</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="#快速开始windows"><img alt="Windows, Linux, and macOS" src="https://img.shields.io/badge/Platforms-Windows%20%7C%20Linux%20%7C%20macOS-555555"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img alt="CPU or CUDA" src="https://img.shields.io/badge/Compute-CPU%20%7C%20CUDA-76B900?logo=nvidia&logoColor=white"></a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <strong>简体中文</strong>
</p>

<p align="center">
  <a href="#功能概览">功能概览</a> ·
  <a href="#快速开始windows">快速开始</a> ·
  <a href="#仿真助手">仿真助手</a> ·
  <a href="#codex-插件入口">Codex 插件</a> ·
  <a href="#开发与验证">开发验证</a>
</p>

<hr>

在本机导入 STL / STEP 模型，设置材料、热源和散热条件，计算并查看三维温度变化。项目包含 Python 仿真服务、浏览器操作界面，以及用于启动服务的 Codex 插件源码。

核心计算采用 Gmsh 四面体网格、scikit-fem 有限元组装和 SciPy / CuPy 求解。适合固体导热算例与交互演示；高级物理功能的适用范围见下方“物理模型与已知限制”。

## 功能概览

| 模块 | 当前功能 |
| --- | --- |
| 模型与材料 | STL、STEP / STP 导入，几何质量检查，材料预设、自定义物性、方框材料分区与组件材料设置 |
| 热源与边界 | 面热源、点热源、嵌入位置，鼠标刷选与按朝向选面，多热源、启停时间、功率曲线、温控、对流与辐射 |
| 三维求解 | 瞬态与稳态固体导热，CPU / CUDA 求解，相变等效热容、等效接触热阻和空气间隙传热 |
| 结果 | 温度动画、最高/最低/平均温度、轴向剖切、热膨胀位移估算、CSV、Markdown / PDF 报告和 ParaView 数据 |
| 算例管理 | 保存与加载配置、运行记录、计算进度与取消 |
| 仿真助手 | 本地规则解析，或通过 Codex CLI / Responses API 生成可检查的配置草案 |
| 辅助分析 | CPU / 主板 / 散热器热网络、换热系数反推、安全功率估算、热泵与制冷循环 COP 估算 |

## 快速开始（Windows）

准备 Python 3.11 或更新版本，以及支持 WebGL 的浏览器。使用 Git 获取源码后，在 **仓库根目录**运行以下命令：

```powershell
git clone https://github.com/tianyuzong/hot_sim_new.git
cd hot_sim_new
powershell -NoProfile -ExecutionPolicy Bypass -File .\Install.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start.ps1 -Device auto
```

`Install.ps1` 创建 `.venv` 并安装 [requirements.txt](requirements.txt) 中的依赖；首次安装需要联网，依赖中包含 CUDA 相关包。`-Device auto` 在 GPU 不可用时回退到 CPU，CPU 运行不要求有 NVIDIA 显卡。

启动后打开 [本地仿真界面](http://127.0.0.1:8765/)。服务只监听本机，Three.js 等页面依赖已保存在 `static/vendor/`，常规本地仿真不依赖 CDN 或 Codex。

### 启动、停止与计算设备

```powershell
# 强制使用 CPU
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start.ps1 -Device cpu

# 要求可用的 NVIDIA CUDA 环境，检测失败时直接报错
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start.ps1 -Device cuda

# 更换端口，并禁止自动打开浏览器
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start.ps1 -Device auto -Port 8766 -NoBrowser

# 查看默认端口服务的实际计算设备与状态
Invoke-RestMethod http://127.0.0.1:8765/api/health

# 停止服务及正在进行的计算；自定义端口时传入对应 -Port
powershell -NoProfile -ExecutionPolicy Bypass -File .\Stop.ps1 -Port 8765
```

| 启动入口 | 默认设备行为 |
| --- | --- |
| `Start.bat` 或未传 `-Device` 的根目录 `Start.ps1` | `cuda`，CUDA 不可用时启动失败 |
| `plugins/thermal-studio/scripts/start.ps1` | `auto`，CUDA 不可用时回退 CPU |
| 直接运行 Python 服务 | 读取 `THERMAL_DEVICE`；未设置时为 `auto` |

启动脚本会用 `-Device` 的值覆盖 `THERMAL_DEVICE`，因此通过脚本切换设备时应明确传参。CUDA 加速稀疏线性求解；模型处理、网格生成、有限元组装和导出仍在 CPU 上执行。

关闭网页不会停止服务。修改设备或环境变量后，需要先停止已有服务，再从设置了新变量的终端启动。日志位于 `.runtime/server.log` 和 `.runtime/server-error.log`。

## 第一个三维算例

1. **导入模型。** 上传封闭的 STL 或 STEP / STP。STL 需指定坐标单位，STEP 按文件声明单位解释；额外尺寸倍率会改变最终大小。导入后先检查实际尺寸。
2. **设置材料。** 选择基础材料；如需多材料，可添加方框区域。预设物性是示例值，应按材料牌号和温度调整。
3. **添加热源。** 面热源必须刷选或按朝向选取受热面；点热源必须有有效位置。填写总功率及开始、结束时间，仅设置材料和计算时长不会产生热源。
4. **设置散热。** 配置环境温度与换热系数 `h`，或创建指定散热区。`h=0` 表示不施加该项对流；热源面是否使用默认散热由相应选项控制。
5. **配置计算。** 设置初始温度、时长、时间步长、保存间隔和网格尺寸，运行仿真。
6. **检查结果。** 查看温度动画、剖面、材料体积、热源映射和能量检查，再按需要导出或保存算例。

界面字段以各自显示的单位为准；API 中带 `_m` 的坐标、半径和网格尺寸使用米，功率使用 W，时间使用 s，温度使用 °C。仓库不包含本机 `data/` 中的模型，示例模型的尺寸与质量取决于实际导入设置。

本地页面入口：[三维仿真](http://127.0.0.1:8765/) · [CPU 热仿真](http://127.0.0.1:8765/cpu.html) · [界面使用说明](http://127.0.0.1:8765/guide.html)。以上链接需要服务已启动；使用其他端口时相应修改地址。

## 仿真助手

先加载一个模型，再打开“智能助手”。生成配置只创建草案；选择“应用后修改”可回到主界面检查参数和选区，选择“确认并仿真”才会提交计算。

### 解析模式与连接方式

| 页面模式 / 配置 | 行为 |
| --- | --- |
| 本地规则解析器 | 无需联网，按内置规则识别参数；复杂请求需检查实际生成内容 |
| Codex，`THERMAL_CODEX_PROVIDER=cli`（默认） | 调用本机官方 `codex exec`，沿用 CLI 已有登录态；找不到 CLI 时明确报错 |
| Codex，`THERMAL_CODEX_PROVIDER=api` | 使用 OpenAI Responses API，需要 `OPENAI_API_KEY` |
| Codex，`THERMAL_CODEX_PROVIDER=auto` | 优先 CLI，仅在找不到 CLI 时改用 API；CLI 请求失败不会自动切换 API |

CLI 模式需要安装并登录官方 Codex CLI，可在终端通过 `codex --version` 和 `codex login status` 检查。程序也会搜索 Windows 上官方 CLI 的本地安装目录，可用 `THERMAL_CODEX_COMMAND` 指定可执行文件路径。调用使用临时会话和只读沙箱。

使用 Codex 时，自然语言目标、当前配置和模型几何摘要会交给所选服务处理；三维仿真计算仍在本机执行。Codex 连接失败会显示错误，不会静默切换到本地规则解析器。

### 包含热源的指令示例

在已经导入模型的页面选择 **Codex**，输入：

```text
基于当前已加载模型创建一个瞬态导热算例，保留当前网格尺寸。
基础材料使用铜，初始温度 25°C，环境温度 25°C。
热源列表仅保留一个名为“顶部加热”的面热源：总功率 20 W，
从 0 秒持续到 600 秒，选择模型顶部（+Z）的外表面作为受热面。
全部外表面设置对流散热，换热系数 10 W/(m²·K)，包括热源面。
仿真总时长 600 秒，时间步长 5 秒，每 5 秒保存一帧。
不使用功率曲线、温控、辐射、相变或额外接触热阻。
如果顶部没有可用受热面，请提示我重新选择，不要生成没有热源的配置。
```

草案应包含 **20 W 热源、0–600 s 启停时间和非零受热面数**。确认前通过“应用后修改”检查选区，尤其要核对模型摆放方向和尺寸。

“顶部/底部、左/右、前/后、全部外表面”会由本地几何代码转换为真实面编号。轴向选区取该方向最外侧 3% 范围内、法向朝向该方向的外表面三角面；它并不代表任意形状完整的上半部分。缺少热源、选区为空、面编号无效或模型不匹配时，草案不能确认运行。

### 环境变量

在启动服务前设置；服务已运行时，设置后需重启才能生效。

| 变量 | 用途与默认值 |
| --- | --- |
| `THERMAL_CODEX_PROVIDER` | `cli` / `api` / `auto`，默认 `cli` |
| `THERMAL_CODEX_COMMAND` | 可选的 Codex CLI 可执行文件路径 |
| `THERMAL_CODEX_MODEL` | 可选的 CLI 模型；不设置时沿用 CLI 配置 |
| `THERMAL_CODEX_TIMEOUT_S` | CLI 等待上限，默认 `180` 秒，允许 `15–600`；不控制 API 超时 |
| `OPENAI_API_KEY` | API 模式必需的密钥；无需写入仓库 |
| `OPENAI_MODEL` | API 模型，代码默认 `gpt-5.2` |
| `OPENAI_BASE_URL` | API 服务根地址，默认 `https://api.openai.com/v1` |
| `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` | 显式代理设置，CLI 子进程优先沿用 |
| `THERMAL_DEVICE` | 直接启动 Python 时的 `auto` / `cpu` / `cuda`；启动脚本会覆盖此值 |
| `THERMAL_CUDA_DEVICE` | CUDA 设备编号，默认 `0` |
| `THERMAL_STUDIO_ROOT` | 插件被放到仓库外时，指定完整应用根目录 |

例如，延长 CLI 等待上限后以 CPU 启动：

```powershell
$env:THERMAL_CODEX_PROVIDER = 'cli'
$env:THERMAL_CODEX_TIMEOUT_S = '300'
powershell -NoProfile -ExecutionPolicy Bypass -File .\Start.ps1 -Device cpu
```

未显式配置代理变量时，CLI 子进程在 Windows 上读取已启用的系统手动代理，并将本机地址加入直连范围；不会修改系统或其他进程的代理。仅配置 PAC 脚本的环境需要另行提供可用的代理环境变量。页面显示等待时长，并区分启动失败、超时和配置校验失败。

## Codex 插件入口

插件源码位于 [plugins/thermal-studio](plugins/thermal-studio/README.md)，包含 `.codex-plugin/plugin.json`、技能说明和启动 / 健康检查脚本。它调用本仓库的服务，不打包 Python 环境、模型或计算结果；克隆仓库、运行 `Install.ps1` 不会自动在 Codex 中安装插件。

完成根目录的依赖安装后，可以在 **仓库根目录**直接使用插件脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\plugins\thermal-studio\scripts\start.ps1 -Device auto
powershell -NoProfile -ExecutionPolicy Bypass -File .\plugins\thermal-studio\scripts\health-check.ps1
```

插件启动脚本支持 `-Port`、`-Device` 和 `-NoBrowser`。独立放置插件时的路径配置见 [插件 README](plugins/thermal-studio/README.md)。

## 保存、导出与备份

| 目录 / 文件 | 内容 |
| --- | --- |
| `data/models/` | 上传模型、导入元数据与表面几何 |
| `data/projects/` | 保存的算例参数与选区快照 |
| `data/jobs/` | 各次计算的输入、日志、网格、温度场及导出结果 |
| `.runtime/` | 服务日志、进程信息与临时运行文件 |
| `.venv/` | 本机 Python 虚拟环境 |

这些目录由 [.gitignore](.gitignore) 排除。**推送源码到 GitHub 不会备份模型、保存的算例或计算结果**；迁移时如需保留它们，应另行备份 `data/`，并在新电脑重新安装依赖。

完整三维结果 ZIP 解压后，在 ParaView 打开 `temperature.xdmf`，将 `thermal-fields.h5` 保留在同一目录，选择 `Temperature_C` 着色。坐标为 m，时间为 s，温度为 °C。`Material_ID` 对应结果中的材料编号；`audit.json` 记录材料体积、热源映射、计算设备与数值检查信息。

## 物理模型与已知限制

三维瞬态求解采用 P1 四面体有限元、一致质量矩阵和后向欧拉时间积分。对流边界输入的是给定换热系数，程序不求解实际风速、流场或流体压力。STEP 方框材料区域参与 CAD 分割；STL 方框区域按单元中心分配，边界精度受网格影响。

以下限制在当前实现中仍需注意：

| 功能 | 当前限制 |
| --- | --- |
| STEP 多个独立实体 | 组件编号映射存在已知问题，可能将多个实体归入同一组件，影响按组件赋材与空气间隙传热；不能仅凭界面材料设置判断分配正确 |
| 功率曲线与相变 | 功率曲线折点尚未自动加入时间网格；跨越折点可能产生输入能量误差。相变按上一时刻温度更新等效热容，跨越相变温区可能触发能量检查失败 |
| 接触热阻与空气间隙 | 接触热阻以材料导热率修正近似，具有网格依赖；空气间隙使用简化几何配对与 `k·A/间距`，未完整处理遮挡、朝向和流动 |
| 热膨胀位移 | 为相对中心的自由膨胀估算，不包含机械约束、应力或完整热弹性求解 |
| 稳态结果审计 | 稳态报告部分能量字段仍沿用瞬态命名，单位与收敛判定有待完善，不能将其直接作为瞬态能量审计使用 |
| CPU 风冷 / 水冷工作台 | 属于集中参数热网络；当前水流量单位换算和显式时间积分稳定性存在已知问题，大步长可能产生异常温度，定量结果需修正与复核 |
| 换热反推与循环估算 | 基于集中参数或理想温差模型；未接入完整工质物性和设备性能曲线 |

开始验证时可使用单一材料、恒定功率和对流边界。程序尚未自动完成网格 / 时间步长收敛分析；应比较不同离散设置，并核对实际尺寸、材料体积、总输入功率和关键温度。能量残差较小本身不能证明物理模型与局部温度正确。

## 开发与验证

从仓库根目录运行：

```powershell
# 开发测试依赖，不是启动服务的必要依赖
.\.venv\Scripts\python.exe -m pip install pytest httpx

# Python 回归测试，无需 CUDA
$env:THERMAL_DEVICE = 'cpu'
.\.venv\Scripts\python.exe -m pytest tests -q

# 前端测试需要本机 Node.js
node --test tests/test_agent_ui.cjs
node --check static/app.js
```

测试覆盖基础物理算例、几何质量、扩展配置、CPU 热网络以及 Agent 连接、热源选区和前端草案校验；它们不代表上述所有高级模型均已完成工程验证。

另有可选的 API 集成脚本：先在默认 `8765` 端口启动服务，再运行：

```powershell
.\.venv\Scripts\python.exe tests/integration_api.py
```

该脚本会创建测试模型和算例、提交实际计算、检查导出，并重写 `validation-api.json`。仓库内 [VALIDATION.md](VALIDATION.md)、`validation-api.json` 和 `validation-wukong.json` 是历史验证记录，其模型与参数不代表当前本机导入状态。

### 源码导航

| 路径 | 职责 |
| --- | --- |
| [server.py](server.py) | 本地 API、模型上传、算例保存、任务与下载接口 |
| [geometry.py](geometry.py) | 几何处理、质量检查、CAD 分区与体积网格 |
| [solver.py](solver.py) | 有限元组装、时间积分、边界条件、剖切、位移估算与导出 |
| [analysis.py](analysis.py) | CPU 热网络、换热系数拟合、功率与循环估算 |
| [agent.py](agent.py) | 自然语言解析、CLI / API 连接、面选区映射与配置校验 |
| [schemas.py](schemas.py) | 参数模型、校验与材料预设 |
| [runtime.py](runtime.py)、[worker.py](worker.py) | 运行目录、工作进程与计算任务执行 |
| [static/](static/) | 三维界面、CPU 工作台、使用说明和本地前端依赖 |
| [tests/](tests/) | Python、前端与 API 集成验证 |
| [plugins/thermal-studio/](plugins/thermal-studio/README.md) | Codex 插件源码与辅助脚本 |

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 双击 `Start.bat` 提示 CUDA / CuPy / DLL 不可用 | 在仓库根目录改用 `Start.ps1 -Device auto` 或 `-Device cpu`；如需 GPU，再检查对应 CUDA 环境 |
| 启动时报找不到 `solver` 模块 | 先切换到仓库根目录，再执行启动脚本 |
| 页面打不开或端口被占用 | 查看 `.runtime/server-error.log`，检查 `/api/health`，或用 `-Port 8766` 启动 |
| Codex 找不到 CLI 或无法连接 | 检查 CLI 安装、登录态和代理；API 使用者需明确设置 provider 与密钥 |
| Codex 等待超时 | 先核对连接及代理，再按需要调整 `THERMAL_CODEX_TIMEOUT_S`；延长等待本身不会修复连接失败 |
| “请先添加至少一个热源” | 检查草案是否真的创建了热源，以及面热源的选面数或点热源位置；可使用上面的完整示例 |
| 模型尺寸、质量或温升异常 | 核对 STL 单位、STEP 文件单位和额外倍率，并检查材料物性、区域体积与热源选区 |

依赖版本见 [requirements.txt](requirements.txt)，第三方组件声明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
