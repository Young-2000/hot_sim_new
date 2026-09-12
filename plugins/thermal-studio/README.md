# Thermal Studio Codex Plugin

此目录包含 Thermal Studio 的 Codex 插件源码，通过技能说明和 PowerShell 脚本调用完整工程根目录的 `Start.ps1`。插件不复制 Python 环境、模型缓存或计算结果；应用安装步骤和功能说明见 [项目 README](../../README.md)。

## 启动与停止

先在完整工程中运行 `Install.ps1` 安装依赖，再从 **工程根目录**运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\plugins\thermal-studio\scripts\start.ps1 -Device auto
powershell -NoProfile -ExecutionPolicy Bypass -File .\plugins\thermal-studio\scripts\health-check.ps1
```

插件启动脚本默认使用 `auto`，CUDA 不可用时回退到 CPU；可通过 `-Device cpu` 或 `-Device cuda` 强制选择。默认地址为 [本地界面](http://127.0.0.1:8765/)，`-Port 8766` 可更换端口，`-NoBrowser` 可禁止自动打开浏览器。

关闭页面不会停止服务。停止服务及当前计算：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Stop.ps1 -Port 8765
```

根目录的 `Start.bat` / `Start.ps1` 默认要求 CUDA，与插件脚本的 `auto` 默认值不同。

## 仿真助手

页面可选择本地规则解析器或 Codex。Codex 默认通过本机官方 `codex exec` 使用已有 CLI 登录态；连接方式由 `THERMAL_CODEX_PROVIDER` 决定：

- `cli`（默认）：调用 CLI，找不到时明确报错。
- `api`：使用 Responses API，需要 `OPENAI_API_KEY`。
- `auto`：优先 CLI，仅在找不到 CLI 时改用 API；CLI 请求失败不会切换 API。

可用 `THERMAL_CODEX_MODEL` 指定 CLI 模型，`THERMAL_CODEX_TIMEOUT_S` 设置 CLI 等待上限（默认 180 秒，范围 15–600 秒）。环境变量需在启动前设置；已有服务需重启。代理处理、完整配置表及带热源的指令示例见 [项目 README](../../README.md)。

生成配置后先检查草案，确认热源有有效受热面或点位置。选择“应用后修改”可返回主界面查看和调整；“确认并仿真”会提交计算。

## 在仓库外放置插件

插件目录独立放置时，设置完整工程路径，并切换到该目录后再调用插件启动脚本：

```powershell
$env:THERMAL_STUDIO_ROOT = 'D:\path\to\hot_sim_new'
Set-Location -LiteralPath $env:THERMAL_STUDIO_ROOT
```

工程必须包含 `Start.ps1`、`server.py`、`static/` 及已安装的 Python 依赖。使用独立插件所在位置的 `scripts/start.ps1` 启动即可。

此目录包含 `.codex-plugin/plugin.json`、`skills/` 和 `scripts/`；克隆仓库或运行应用的 `Install.ps1` 不会自动将插件安装到 Codex。
