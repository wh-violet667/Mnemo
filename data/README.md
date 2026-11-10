# 本地评测语料

`data/violet-evergarden-kb/` 放的是 violet 50 评测使用的本地语料。它默认被 `.gitignore` 忽略，不提交到公开仓库。

## 目录结构

```text
data/violet-evergarden-kb/
├── README.md                       # 语料说明
├── docs/                           # 12 份 Markdown 主题文档
├── docx/                           # 12 份评测 DOCX + README.docx
└── data/
    └── qa-benchmark.zh.jsonl       # 50 条问题、参考答案和标签
```

`README.docx` 只是语料说明，上传脚本会自动排除；真正参与评测的是 `docx/` 下的 12 份内容文档。

## 来源与许可

这份语料是《紫罗兰永恒花园》的非官方粉丝整理，仅用于技术测试、学习与研究。作品版权归晓佳奈、京都动画及相关权利方所有；语料没有提供再分发许可证。

因此：

- 不要把完整语料提交到公开 GitHub 仓库；
- 课程提交包可以自带这份本地语料；
- 公开仓库只保留 `golden_violet.jsonl`、脚本和说明。

## 使用方式

默认路径：

```bash
python evals/seed_violet_kb.py --check-only
python evals/seed_violet_kb.py --base-url http://127.0.0.1:8010
python evals/build_violet_golden.py
```

也可以自定义路径：

```bash
python evals/seed_violet_kb.py --corpus-dir data/violet-evergarden-kb
python evals/build_violet_golden.py --kb-dir data/violet-evergarden-kb
```

或设置环境变量：

```bash
VIOLET_KB_DIR=data/violet-evergarden-kb
```
