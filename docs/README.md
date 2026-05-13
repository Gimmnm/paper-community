# 文档索引

除 **`experiment_report/`** 子目录（LaTeX 报告）外，**说明类 Markdown 均平铺在 `docs/` 根目录**，便于查找。

---

## 三本核心（建议先读）

| # | 文档 | 给谁 | 内容 |
|---|------|------|------|
| 1 | [`user_manual_zh.md`](user_manual_zh.md) | 用网站的人 | Web 布局、两种模式、检索与图、Eval、环境与启动摘要 |
| 2 | [`developer_manual_zh.md`](developer_manual_zh.md) | 开发与运维 | 总流程、模块与数据流、`demo-api`、API/前端要点、命令与脚本、缓存与典型字段 |
| 3 | [`offline-outputs-catalog.md`](offline-outputs-catalog.md) | 所有人（查路径时） | **`out/` / `data/`** 各目录与文件：含义、格式、生成命令、推荐跑数顺序 |

另：**[`../README.md`](../README.md)**（仓库门面）、**[`../src/README.md`](../src/README.md)**（`src/` 分层与 `core.py` 入口）。

---

## 同目录补充规格（仍有参考价值）

| 文档 | 用途 |
|------|------|
| [`requirements-breakdown.md`](requirements-breakdown.md) | 产品目标与三层分工陈述 |
| [`architecture-boundaries.md`](architecture-boundaries.md) | 模块边界 |
| [`algorithm-pipeline-spec.md`](algorithm-pipeline-spec.md) | 管线契约 |
| [`evaluation-metrics-spec.md`](evaluation-metrics-spec.md) | 评估指标口径 |
| [`experiment-comparison-pipeline.md`](experiment-comparison-pipeline.md) | 多算法离线对比、manifest、断点表契约 |
| [`out-archive-policy.md`](out-archive-policy.md) | `out/` 归档与 git 策略 |
| [`cleanup-and-validation.md`](cleanup-and-validation.md) | 清理与校验清单 |

---

## 报告与归档

| 类型 | 路径 |
|------|------|
| LaTeX 实验报告 | [`experiment_report/README.md`](experiment_report/README.md) |
| 已冻结的规划长文 | [`../archive/legacy/docs/refactor_archived/README.md`](../archive/legacy/docs/refactor_archived/README.md) |
| Web 早期基线（英文） | [`../archive/legacy/docs/web_archived/`](../archive/legacy/docs/web_archived/) |
| 其它 legacy | [`../archive/legacy/`](../archive/legacy/) |

（`archive/legacy/docs/specs/` 已腾空，见其中 `README.md`。）

---

## 维护原则

- **路径与列名**以 `offline-outputs-catalog.md` 为准；**流程与命令**以 `developer_manual_zh.md` + `python src/core.py <task> --help` 为准；**界面**以 `user_manual_zh.md` 为准。
