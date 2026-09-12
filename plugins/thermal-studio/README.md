# Thermal Studio Codex Plugin

这个插件包装仓库中的 Thermal Studio 本地服务。它不复制 Python 虚拟环境、模型缓存或计算结果，启动时直接调用工程根目录的 `Start.ps1`。

## 启动

在 Codex 终端运行：

```powershell
.\plugins\thermal-studio\scripts\start.ps1
```

脚本默认监听 `http://127.0.0.1:8765/`，计算设备使用 `auto`。需要强制 CPU 或 CUDA 时可执行：

```powershell
.\plugins\thermal-studio\scripts\start.ps1 -Device cpu
.\plugins\thermal-studio\scripts\start.ps1 -Device cuda
```

使用 `-Port 8766` 可避开已占用端口，使用 `-NoBrowser` 可禁止自动打开浏览器。停止服务仍使用工程根目录的 `Stop.ps1`。

## Codex Agent 模式

页面中的 Agent 可以选择本地规则解析器或 Codex。Codex 模式读取当前用户环境变量 `OPENAI_API_KEY`，可选 `OPENAI_MODEL` 和 `OPENAI_BASE_URL`；插件不会保存或打印密钥。提交自然语言配置后，先检查配置草案，再点击“确认并仿真”。

## 独立安装

如果插件目录被复制到其他位置，请设置工程根目录：

```powershell
$env:THERMAL_STUDIO_ROOT = 'D:\path\to\hot_sim_new'
```

根目录必须包含 `Start.ps1`、`server.py` 和 `static/`。完整的物理模型、参数说明和导出格式见工程根目录 `README.md`。
