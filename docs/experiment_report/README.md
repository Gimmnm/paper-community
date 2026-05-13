# 离线实验报告（LaTeX，中文）

本目录的 `main.tex` 为**中文、展开版**技术说明，涵盖四算法社区发现、分辨率诊断、Topic-SCORE、主题塌缩诊断与检索六路/按分辨率汇总等；与 `docs/offline-outputs-catalog.md` 一致。

## 编译要求

- TeX Live / MacTeX 等，且需 **`latexmk` + `xelatex`**（`ctex` 中文排版）。
- 未安装 XeLaTeX 时请先安装完整 TeX 套件。

## 编译

```bash
cd docs/experiment_report && make
```

等价命令：`latexmk -xelatex -interaction=nonstopmode main.tex`

生成 `main.pdf`（及 XeLaTeX 中间文件如 `main.xdv`）。缺图时文中为占位框，仍可完整编译。

**版本库**：`main.pdf` / `*.xdv` 已列入 `.gitignore`，本地编译即可；勿将大体积 PDF 作为必需提交物。

## 检索图 tag

修改 `main.tex` 中的 `\ComparisonRunTag`（默认 `master_breakpoints`），以匹配 `out/comparison_runs/<tag>/` 与 `retrieval_*__<tag>.png` 文件名。
