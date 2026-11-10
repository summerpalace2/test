# MedMiner 医学数据挖掘平台

面向医学数据挖掘课程实验的**桌面分析软件**（Python + PySide6）。
导入数据 → 预处理 → 可视化 → 统计分析 → 回归/分类/聚类 → 可解释性 → AI 解读 → 导出报告，一条龙。

> 仅用于教学与研究，不构成临床诊断或治疗建议。

## 1. 功能一览

| 页面 | 能做什么 |
|---|---|
| 1 数据管理 | 导入 CSV / TSV / **TXT / .data** / **Excel(多工作表)** / JSON；自动识别编码(含 GBK)、分隔符、表头；`?` 等按缺失值处理；列类型/中文说明可改；指定目标列；数据质量与临床范围检查；项目文件 `.medproj`；拖拽打开 |
| 2 数据预处理 | 15 种步骤：删列/去重/缺失行列、**纳入排除标准筛选**、临床范围处理、异常值(IQR/3σ/MAD/孤立森林)、二值化、取值映射、类型转换、变换、**缺失填充(含 KNN)**、编码、缩放、特征筛选。步骤可停用/调序/保存为 JSON 复用 |
| 3 数据可视化 | 15 种图：直方/密度/箱线/小提琴/蜂群/散点/回归线/计数/构成比/均值±CI/饼/折线/相关热图/散点矩阵/缺失分布 |
| 4 统计分析 | 描述统计、正态性检验、**基线特征表 Table 1**(自动 t/Mann-Whitney/ANOVA/Kruskal/卡方/Fisher)、相关性、**逻辑回归 OR + 森林图** |
| 5 建模评估 | 分类 11 种、回归 12 种算法；**重复分层 K 折交叉验证**；灵敏度/特异度/AUC/PR/校准/DCA；**阈值滑块**(约登最优、灵敏度≥90%)；bootstrap 95% CI；置换重要性、SHAP；自动调参；保存模型 |
| 6 聚类分析 | K-Means / 层次 / GMM / DBSCAN / BIRCH；肘部+轮廓选 K；PCA/t-SNE；簇画像热图；与真实标签对照(ARI/NMI) |
| 7 预测 | 输入一个样本 → 风险概率 + SHAP 主要影响因素；批量预测文件；载入已保存模型 |
| 8 图像数据 | 浏览图像文件夹(子文件夹=标签)，提取 26 个图像特征并**转成数据集**，之后可用以上全部功能 |
| 9 AI 分析 | 接入大模型解读：数据画像、预处理建议、统计、模型、聚类、单个样本、报告结论；流式输出、可追问 |
| 10 报告导出 | 各页“加入报告”或**一键生成完整报告**；导出 HTML / Word / PDF / Markdown |

## 2. 设计上专门为医学小样本做的事

- **默认在交叉验证每一折内部拟合预处理**（填充/缩放/独热），避免数据泄漏；预处理页的“拟合类”步骤会明确标注。
- **数据泄漏检测**：目标由某列二值化得来却把原列留在特征里（如 `num`→`dx`），会自动取消勾选并在训练前警告。实测这种泄漏会把 AUC 虚高到 ≈1。
- 报告 **均值±标准差** 与 **bootstrap 95% 置信区间**；样本量 n<500 会反复提示结果不稳定。
- 医学指标优先：灵敏度(漏诊)、特异度(误诊)、PPV/NPV、可调阈值、决策曲线(DCA)。
- 置换重要性在**测试折上**计算，而不是训练集。

## 3. 安装与运行

需要 Python 3.10–3.13（Python 3.14 部分依赖可能还没有对应安装包，请不要用）。

**Windows 最省事：双击 `start_windows.bat`。** 它会自动创建虚拟环境、安装依赖（默认走清华镜像，失败回退官方源）并启动软件；第一次需要几分钟，之后再双击就是秒开。不想用 exe（解压 3GB 容易出问题）时推荐这种方式。

手动方式：

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
python run.py            # 或 python -m medminer
```

首次体验：打开软件 → “加载示例数据”。`resources/sample/` 里有 csv / 无表头 txt / xlsx 三种格式和一个图像示例文件夹。

> **重要：示例数据是模拟的。** 它的结构与 UCI Cleveland 心脏病数据集一致(303×14，含 `?` 缺失)，但数值是随机生成的，且比真实数据更容易区分(AUC≈0.97，真实数据通常≈0.9)。做实验请用你们自己的 `processed.cleveland.csv`。

## 4. 推荐使用流程（以心脏病数据为例）

1. **数据管理**：打开 `processed.cleveland.csv`；确认 `cp/restecg/slope/ca/thal` 是“分类”；目标列 `num`。
2. **数据预处理**：添加“二值化”（`num` 大于 0 → 新列 `dx`）；回数据管理把目标列改为 `dx`。
3. **统计分析**：按 `num` 或 `dx` 生成基线表；逻辑回归 OR。
4. **建模评估**：目标选 `dx`（`num` 会被自动排除），勾选 逻辑回归/随机森林/XGBoost → 开始训练 → 看对比图、ROC、阈值；算 SHAP。
5. **聚类**：不含诊断列做 K-Means(K=2~4)，看簇画像，与 `num` 对照。
6. **AI 分析**：设置里填 API Key → 一键解读各步结果。
7. **报告导出**：一键生成完整报告 → 导出 Word/PDF。

## 5. AI 模块

- **多套模型配置**：在“AI 分析”页右上角点 **管理模型…**，每个服务商各存一套（服务商 + Base URL + 模型 + 你自己的 API Key），之后在顶部下拉框一键切换，互不覆盖。
- 内置预设：**Claude (Anthropic)**、**OpenAI**、**智谱 GLM**、**DeepSeek**、通义千问、Moonshot、Google Gemini(OpenAI 兼容)、硅基流动、OpenRouter、**本地 Ollama**，以及 **自定义 (OpenAI 兼容 / Anthropic 兼容)**——任何中转/代理地址，选“自定义”填上 Base URL 和 Key 即可。
- 点 **获取模型列表** 会向该服务商查询真实可用的模型；也可直接手填模型名。点 **测试连接** 验证 Key 和地址是否可用。
- 使用的是“OpenAI 兼容”接口时，Base URL 填到 `/v1`（或服务商文档里的对应前缀）为止，软件会自动拼上 `/chat/completions`；Anthropic 协议拼 `/v1/messages`。
- Key 也可用环境变量：`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` / `DASHSCOPE_API_KEY` …
- **隐私**：只发送统计摘要（均值、分布、指标表…），不发送逐行数据；文本列内容永不发送；默认**隐藏列名**(V1、V2…，回答里自动还原，**包括你提问里写的列名**)；默认**每次发送前弹出预览**，可看到将要发出的全部内容。用本地 Ollama 则数据完全不出本机。
- Key 保存在 `~/.medminer/config.json`（Linux/macOS 权限 600；Windows 为普通用户文件），不会写进项目文件或报告。
- 提示词要求模型：只依据摘要作答、不得编造数字、小样本要提示风险、结尾声明非临床建议。**仍请自行核对回答里的数字。**

## 6. 打包成 exe（Windows）

在 Windows 上双击 `build_windows.bat`（会建虚拟环境、装依赖、调用 PyInstaller），得到 `dist\MedMiner\MedMiner.exe`，把整个 `dist\MedMiner` 文件夹拷给别人即可运行，无需安装 Python。体积约 1GB 级(含 scikit-learn/SHAP/XGBoost/Qt)。

**验证打包产物**：命令行运行 `MedMiner.exe --selftest`（无界面），会依次检查：读取 csv/txt/xlsx → 训练 5 种模型(含 XGBoost/LightGBM) → SHAP → 图表 → 聚类 → 导出 HTML/Word/Markdown/PDF → 构造 10 个页面，结果写入当前目录的 `selftest_result.txt`，末尾出现 `SELFTEST OK` 即完整。源码方式同样可用：`python run.py --selftest`。

## 7. 目录结构与 5 人分工建议

```
medminer/
├─ run.py                     启动入口
├─ medminer/
│  ├─ core/                   纯 Python 算法层（无界面，可单独测试）
│  │   io.py schema.py dataset.py medical.py sample_data.py   数据
│  │   preprocess.py stats.py plots.py                         预处理/统计/图表
│  │   models.py features.py evaluate.py explain.py cluster.py 建模
│  │   persist.py images.py report.py                          模型保存与泄漏检测/图像/报告
│  ├─ ai/                     config.py client.py context.py prompts.py
│  └─ gui/                    state.py main_window.py theme.py + pages/ + widgets/
├─ tests/                     123 个测试（pytest）
└─ resources/sample/          示例数据
```

| 成员 | 负责 | 主要文件 |
|---|---|---|
| A 组长 | 主框架、数据导入、打包、集成 | `gui/main_window.py` `gui/pages/data_page.py` `core/io.py` `core/dataset.py` |
| B | 预处理、统计、医学数据清洗 | `core/preprocess.py` `core/stats.py` `gui/pages/preprocess_page.py` `stats_page.py` |
| C | 可视化、图像数据 | `core/plots.py` `core/images.py` `gui/pages/visualize_page.py` `image_page.py` |
| D | 算法、评估、SHAP、聚类 | `core/models.py` `evaluate.py` `explain.py` `cluster.py` `gui/pages/model_page.py` `cluster_page.py` |
| E | AI、报告、预测面板 | `ai/` `core/report.py` `gui/pages/ai_page.py` `report_page.py` `predict_page.py` |

**扩展算法**只需在 `core/models.py` 的注册表里加一项（名称、参数、构造函数），界面参数表单自动生成。

## 8. 测试

```bash
pip install pytest
python -m pytest tests -q          # 约 1.5 分钟；无显示器环境会自动用 offscreen
```

测试覆盖：文件读取(含 GBK/空白分隔/无表头)、预处理每一步、统计检验、全部分类/回归/聚类算法、SHAP、保存/载入、数据泄漏检测、全部图表、AI 客户端(本地假服务，含隐私断言)、报告导出(重新打开 docx/pdf 校验)、整条界面流程。

## 9. 已知局限（请如实写进实验报告）

- 所有测试使用**模拟数据**与**本地假大模型服务**；**未**用真实 Cleveland 数据和真实 API 做过验证。
- 打包配置已在 **Linux 上构建并通过 `--selftest`**；**Windows 的 exe 未实际构建/运行过**（开发环境为 Linux），请先在 Windows 上跑 `build_windows.bat` 再 `MedMiner.exe --selftest`。
- 界面在无显示器环境（offscreen）下渲染截图并走通全流程，**没有人工在真实桌面上点过**，版面/DPI 细节请自行过一遍。
- SHAP 目前支持二分类与回归；多分类请用置换重要性。
- 调参使用同一份数据的内层交叉验证，结果偏乐观，调参后应重新做交叉验证对比。
- 基线表未做多重比较校正。
- DICOM 等专用医学影像格式未直接支持，请先导出为 PNG/TIFF；图像特征为手工统计特征，不是深度学习。
- 模型文件(`.joblib`)载入时可执行其中代码，只载入自己生成的文件。
