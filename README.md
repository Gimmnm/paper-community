# Paper Community Analysis Pipeline

围绕 **ingest → 向量与图 → 多分辨率社区 → 评测与 Web** 的一条数据流；命令行总入口为：

```bash
PYTHONPATH=src python src/core.py <task> [args...]
```

---

## 文档（三本核心 + 索引）

| 文档 | 内容 |
|------|------|
| [`docs/README.md`](docs/README.md) | **文档索引**：三本核心 + 同目录补充规格（`requirements-breakdown`、`experiment-comparison-pipeline` 等） |
| [`docs/user_manual_zh.md`](docs/user_manual_zh.md) | **① 网站用户手册**（中文） |
| [`docs/developer_manual_zh.md`](docs/developer_manual_zh.md) | **② 开发者技术手册**：总流程、代码与模块、数据流、命令与脚本、`demo-api`、实现要点 |
| [`docs/offline-outputs-catalog.md`](docs/offline-outputs-catalog.md) | **③ 输出数据目录**：`out/` / `data/` 路径、格式与生成命令 |
| [`src/README.md`](src/README.md) | `src/` 分层与 CLI 入口（一分钟） |

历史版根 README 中的长篇任务说明已并入 **开发者手册**（§7、§11–§19）；参数与示例以 **`python src/core.py <task> --help`** 为准。

---

## 最短上手

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-api.txt
```

启动 Web 演示（需已有 sweep / 图等产物；完整参数见开发者手册 §7）：

```bash
PYTHONPATH=src python src/core.py demo-api --host 127.0.0.1 --port 8000
```

浏览器：`http://127.0.0.1:8000/`；OpenAPI：`/docs`。
