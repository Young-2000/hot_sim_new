# Thermal Studio Local

本地三维瞬态热仿真平台，已配置这台电脑可用的运行环境。

双击 **Start.bat**，浏览器打开 **http://127.0.0.1:8765**。关闭网页不会关闭计算服务；需要停止时运行 Stop.ps1。Start.ps1 可使用 `-Port 8766` 指定另一个端口。程序只监听本机，页面静态依赖已保存在 static/vendor，运行时不依赖 CDN。

计算默认使用 CUDA GPU（CuPy 稀疏共轭梯度求解器），网格生成、有限元组装和结果导出仍在 CPU 执行。启动后可通过 `/api/health` 查看 `compute.device` 和 GPU 名称。设置 `THERMAL_DEVICE=cpu` 可强制使用 CPU；设置为 `cuda` 时若 CUDA 不可用会直接报错，`auto`（默认）会自动回退到 CPU。`THERMAL_CUDA_DEVICE` 可选择 CUDA 卡编号。

仿真助手提供两种模式：`local` 使用本地规则解析器，无需联网；`codex` 通过 OpenAI Responses API 生成配置，需先设置 `OPENAI_API_KEY`，可用 `OPENAI_MODEL` 指定模型（默认 `gpt-5.2`），可用 `OPENAI_BASE_URL` 指定兼容 Responses API 的服务地址（默认 `https://api.openai.com/v1`）。Codex 模式请求失败或未配置密钥时会明确提示，不会静默切换到本地模式。

支持 STL/STEP 导入、铜/铝/铁等材料及自定义物性、实体内部方框分区、鼠标刷选和按朝向选取热源/散热面、稳态与瞬态导热、多个热源及功率曲线、温控启停、对流与辐射、相变潜热、等效接触热阻、三维温度播放、任意轴向剖切、保存算例、运行记录、CSV 与 ParaView 导出。另有集中参数换热系数反推、安全功率优化和热泵/制冷循环 COP 估算页面。

详细使用步骤和物理模型说明在平台“使用说明”中。三维求解支持固体导热、指定换热系数的对流和表面辐射；材料相变使用温区等效热容，接触热阻使用等效界面层近似。STEP 方框区域生成与界面对齐的网格，STL 区域按单元中心分配，界面精度受网格影响。热泵/制冷页面是理想温差循环估算，精确工程计算仍需工质物性和设备性能曲线。

目录：

- server.py：本地 API、模型上传、算例保存与工作进程管理。
- geometry.py：STL/STEP 处理、CAD 分区与体积网格。
- solver.py：多材料矩阵、稳态/瞬态积分、功率控制、对流/辐射边界、相变等效热容、剖切与结果导出。
- analysis.py：实测温度换热系数拟合、安全功率反推和制冷循环估算。
- agent.py：本地/Codex 自然语言工况解析、受热面自动选取和配置校验。
- schemas.py：输入校验与材料预设。
- static/：本地三维操作界面及 Three.js。
- data/models/：导入模型与表面几何。
- data/projects/：保存的算例快照。
- data/jobs/：每次计算的输入、日志、体积温度场及导出包。
- tests/test_physics.py：均匀加热解析解、恒温保持、线性场截面及输入范围校验。
- validation-api.json、validation-wukong.json：实际 STL/STEP 双材料与悟空三区的验证记录。

测试：`python -m pytest tests/test_physics.py -q`。pytest 仅用于开发测试，不是运行平台的必要依赖。

迁移到其他电脑：安装 Python 3.11 或更新版本，然后运行 Install.ps1 创建 .venv 并安装 requirements.txt；再运行 Start.bat。大模型需要足够内存。当前平台没有自动进行网格收敛分析，应对关注的工况比较网格与时间步长。

完整结果 ZIP 解压后，在 ParaView 中打开 temperature.xdmf，将 thermal-fields.h5 保留在同一目录，选择 Temperature_C。坐标单位 m，时间 s，温度 °C；Material_ID 对应基础材料及按顺序添加的材料区域。audit.json 保存材料体积、功率映射和能量残差。

内置悟空模型按用户指定的厘米解释数值，高约 99 cm；全铜时质量约 1.67 吨。导入其他 STEP 时遵循文件单位，不会自动采用此额外倍率。
