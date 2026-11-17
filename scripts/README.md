# scripts

这里只保留可复用的项目脚本，不包含任何个人机器路径或管理员安装脚本。

## build_docs_site.py

把 `docs/project-introduction.md` 生成 GitHub Pages 页面 `docs/index.html`。

```bash
python scripts/build_docs_site.py
```

GitHub Pages 从 `main` 分支的 `docs/` 目录发布，因此修改 Markdown 后需要重新生成并提交 `docs/index.html`。

依赖：`Markdown==3.10.2`。

## smoke_agentic.py

手动验证 Agentic-lite 的 planner / critic 是否能返回合法 JSON：

```bash
python scripts/smoke_agentic.py
```

它会读取项目根目录或 `backend/` 下的 `.env`，调用真实的 `SUMMARIZATION_MODEL`，因此：

- 需要有效 API Key 和网络；
- 会产生少量模型调用费用；
- 不建议放进 CI 的每次提交检查，应该按需手动运行。

## Windows / Docker 安装

通用的 WSL2、Docker Desktop、PostgreSQL、Redis 和环境变量步骤见：

[`docs/windows-setup.md`](../docs/windows-setup.md)

仓库不提供自动修改 DISM/WSL 或安装 Docker 的管理员脚本，避免把个人机器配置带进公开项目。
