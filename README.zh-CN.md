# Grove — 柑橘自动标注流水线

[![CI](https://github.com/leeohwang/mandarin_tree_detector/actions/workflows/ci.yml/badge.svg)](https://github.com/leeohwang/mandarin_tree_detector/actions/workflows/ci.yml)

[English](README.md) · **简体中文**

Grove 把一整个文件夹的柑橘树照片转化为**目标检测训练数据**（YOLO + COCO）。
开放词表检测器先画出草稿框，人工在本地浏览器界面中修正，修正后的标签再导出为一份干净的
数据集——这是一座为采摘柑橘机器人的感知模型量产训练标签的「工厂」。

> **交付物是标签数据（坐标文件），而不是带框的图片。** 预览图存在的唯一目的，
> 是让人能核对标签是否正确。完整的构建规格见 `SPEC.md`。

如果你是操作员（而非开发者），你要看的是 **`OPERATOR_GUIDE.md`**，而不是本文件。

---

## 架构一览

Grove 沿着 **GPU 边界** 一分为二。两半之间**只通过磁盘上的文件**（工作目录 / 数据集目录）
通信——正是这一点，让繁重的一半能跑在免费云端 GPU 笔记本上，而轻量的一半能跑在没有 GPU 的
笔记本电脑上。

```
                       ┌──────────────── 共享内核 (grove.core) ────────────────┐
                       │  models.py  规范化 BBox / Detection / ImageRecord /     │
                       │             Manifest（Pydantic）                        │
                       │  config.py  一份经校验的 YAML -> Config                 │
                       │  formats.py 规范格式 <-> YOLO / COCO / 像素 转换器      │
                       │             （无需 GPU、依赖极轻、有单元测试）          │
                       └────────────────────────────────────────────────────────┘
        GPU 侧（Kaggle/Colab）— grove.pipeline          本地侧（Mac）— grove.review
        ────────────────────────────────────────          ──────────────────────────────────
        ingest    文件夹       -> manifest.json           server.py  FastAPI 接口
        detect    manifest     -> predictions.json         store.py   修正（权威来源）
        annotate  predictions  -> 预览图                    static/    canvas 界面（新建/移动/
        export    当前状态     -> 草稿 YOLO + COCO                    缩放/删除/改标签）
        train     已审数据集   -> 学生 YOLO（可选）         -> 重新导出最终 YOLO + COCO
```

**规范坐标格式**（代码内部唯一使用的表示）：归一化 `xyxy`，原点在左上角，每个值都在
`[0, 1]` 区间内。格式转换**只发生在边界处**（`grove/core/formats.py`）。这里是最容易引入
隐蔽 bug 的地方，所以转换器被最先编写并做了单元测试（`tests/test_formats.py`）。

**检测器是「老师」，不是机器人本身。** Grounding DINO（通过 `autodistill`）精度高但速度慢——
它负责**标注**数据。可选的 `train` 阶段会把这些经人工审校的标签蒸馏成一个快速的 YOLO
**学生模型**，而它才是唯一可部署到机器人上的产物。

---

## 端到端数据流

```
原始图片
  -> data/work/manifest.json          （ingest:   扫描 + EXIF 校正 + 稳定 id）
  -> data/work/predictions.json       （detect:   开放词表检测器，可选切片）
  -> data/work/previews/*             （annotate: 为人工 QC 画框）
  -> data/dataset/ （草稿 YOLO+COCO）  （export）
  --- 下载数据集，在本地审校 ---
  -> data/work/review_store.json      （review:   人工修正，与预测分开保存）
  -> data/dataset/ （最终 YOLO+COCO）  （review 重新导出——真正的交付物）
  -> data/work/runs/ （学生 YOLO）     （train，可选）
```

经过 EXIF 校正、准备好的图片存放在 `data/work/images/<id>.<ext>`；每条
`ImageRecord.path` 都**相对于 `work_dir`** 存储，因此数据集可以在云端和笔记本电脑之间自由迁移。

---

## 仓库结构

```
grove/
├── core/                  # 无 GPU 的共享内核（任何地方都可安全导入）
│   ├── models.py          # 规范化 BBox / Detection / ImageRecord / Manifest
│   ├── config.py          # Pydantic 配置 schema + load_config()
│   └── formats.py         # 规范格式 <-> YOLO / COCO / 像素 转换器
├── pipeline/              # GPU 侧
│   ├── ingest.py          # 文件夹 -> Manifest（EXIF 校正、稳定 id）
│   ├── detectors/
│   │   ├── base.py        # Detector 协议 + get_detector() 注册表
│   │   ├── grounding_dino.py
│   │   └── yolo_world.py
│   ├── tiling.py          # 切片 -> 逐片检测 -> 合并（NMS）  [SAHI 策略]
│   ├── detect.py          # 在 manifest 上运行检测器（可断点续跑）
│   ├── annotate.py        # 为 QC 画框（supervision）
│   ├── export.py          # 写出 YOLO + COCO（确定性切分、带校验）
│   └── train.py           # 可选：蒸馏学生 YOLO（ultralytics）
├── review/                # 本地侧（无 GPU）
│   ├── server.py          # FastAPI 应用 + 接口
│   ├── store.py           # 标注存储（修正的唯一真实来源）
│   └── static/            # index.html, app.js, styles.css（canvas 界面）
└── cli.py                 # grove ingest|detect|annotate|export|train|review

notebooks/kaggle_label.ipynb   # 一次 Run-All 的 GPU 任务（安装 -> ... -> 打包数据集）
tests/                          # test_formats（最先）、test_tiling、test_ingest、test_export
config.example.yaml             # 带注释的配置模板（复制为 config.yaml）
setup.sh / Makefile             # 一条命令完成本地安装 + 便捷目标
```

---

## 命令行（CLI）

所有子命令都读取同一份 YAML（`--config config.yaml`，默认 `config.yaml`）：

```
grove ingest    # 文件夹 -> manifest.json
grove detect    # manifest -> predictions.json            （GPU；可续跑）
grove annotate  # predictions -> 预览图                    （GPU）
grove export    # 当前状态 -> YOLO + COCO
grove train     # 已审数据集 -> 学生 YOLO                  （GPU；可选）
grove review    # 启动本地 FastAPI 审校界面                （无 GPU）
```

`grove review` 只导入轻量的 `[review]` 依赖栈，因此可以在没有 GPU 的 Mac 上运行。

---

## 配置

一份 YAML 驱动一切（见 `config.example.yaml`）。操作员通常**只需要改路径和检测器的
ontology（提示词映射）**；其余每个字段都有可用的默认值。

```yaml
detector:
  backend: grounding_dino          # grounding_dino | grounding_dino_hf | yolo_world （可替换，§2.5）
  ontology:
    "tree trunk": tree             # 提示词文本 -> 类别名（默认面向单棵树）
  box_threshold: 0.20
  text_threshold: 0.15
tiling:
  enabled: false                   # 处理大幅画面中又小又远的果实时打开
export:
  formats: [yolo, coco]
  val_split: 0.15
  seed: 42                         # 确定性切分
```

提示词的措辞和 `box_threshold` 对数据集很敏感——调参方法见 `SPEC.md` §8/§11。

---

## 开发

```bash
./setup.sh          # venv + 无 GPU 的 [review,dev] 安装
make test           # 运行无 GPU 的测试套件（转换器、切片、ingest、export）
```

**持续集成（CI）。** `.github/workflows/ci.yml` 会在每次向 `main` 推送或提 PR 时运行完整测试
套件，覆盖 Python 3.10/3.11/3.12。它安装轻量的 `[review,dev]` 依赖栈外加 `supervision`，
因此即便是检测器适配层的测试也能在普通 CPU runner 上跑——GPU 后端的 `supervision -> 规范格式`
转换、注册表 / 构造分发，以及 autodistill -> HF 的回退逻辑全都被覆盖（autodistill 在进程内被 mock）。
只有真正需要 CUDA + 权重的模型前向计算会自行跳过。

- **core** 保持依赖极轻（仅 pydantic + pyyaml），因此在任何机器上都能导入。
- 重度依赖 GPU 的后端（`autodistill`、`supervision`、`ultralytics`）藏在 `[gpu]` extra 之后并
  延迟导入，所以导入 CLI 或审校服务器时绝不会拖进 torch。
- 测试中检测器后端被**打桩（stub）**，因此流水线和导出器无需 GPU 即可测试。

已锁定版本的依赖分组（`pyproject.toml`）：base（轻量）· `[review]`（FastAPI，无 torch）·
`[gpu]`（Kaggle）· `[dev]`（pytest、httpx）。

权威规格、构建里程碑（M0–M5）和设计取舍详见 `SPEC.md`。
