# Windows 开发环境说明

这份文档只描述通用的 Windows / WSL2 / Docker 安装步骤，不包含任何本机盘符、用户名或安装包路径。

## 1. 前置条件

- Windows 10/11，建议开启虚拟化
- Python 3.12+
- Git
- Docker Desktop（使用 WSL2 backend）
- 可访问 OpenCode Go、DashScope 和 Milvus/Zilliz 的网络

## 2. 开启 WSL2

以管理员身份打开 PowerShell，执行：

```powershell
wsl --install
```

如果系统已经安装过 WSL，可以只更新：

```powershell
wsl --update
```

完成后重启 Windows。再次检查：

```powershell
wsl --status
```

## 3. 安装 Docker Desktop

从 Docker 官网下载安装 Docker Desktop，安装时勾选 WSL2 backend。安装完成后启动 Docker Desktop，等待引擎就绪：

```powershell
docker version
docker compose version
```

如果 `docker version` 能同时显示 Client 和 Server，说明引擎已经可用。

## 4. 启动 PostgreSQL 和 Redis

在项目根目录执行：

```powershell
docker compose up -d
docker compose ps
```

当前 `docker-compose.yml` 只启动 PostgreSQL 和 Redis Stack；Milvus 使用 `.env` 中配置的 Zilliz/Milvus 服务。

## 5. 配置环境变量

```powershell
copy .env.docker.example .env
```

至少填写：

- `OPENCODE_API_KEY`：对话模型
- `DASHSCOPE_API_KEY`：embedding / rerank
- `Milvus_url`、`Token`：Milvus/Zilliz 连接

`DATABASE_URL`、`REDIS_URL` 在 `.env.docker.example` 里已按 `docker-compose.yml` 的 `mnemo:mnemo` 默认值填好；如果你改了 compose 配置，再同步修改这两项。`.env.example` 是手工部署/自定义环境的模板，里面的 `user:pass` 是占位符，不能直接用于 docker compose。

不要提交 `.env`。仓库只保留 `.env.example` / `.env.docker.example`。

## 6. 启动后端

```powershell
cd backend
$env:PORT="8010"
..\.venv\Scripts\python.exe main.py
```

如果 8000 端口被占用，使用 8010。后端启动后会初始化数据库表、Milvus collection 和 Redis 连接。

## 7. 启动前端

回到项目根目录：

```powershell
$env:MNEMO_API_BASE_URL="http://127.0.0.1:8010"
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py --server.port 8501
```

打开 `http://localhost:8501`。

## 8. 常见问题

### Docker 引擎未启动

先打开 Docker Desktop，确认托盘图标显示运行中，再执行 `docker version`。

### WSL 未就绪

以管理员身份执行：

```powershell
wsl --install --no-distribution
wsl --update
```

然后重启。

### 端口冲突

- FastAPI 默认 8000，可用 `PORT=8010` 改为 8010。
- Streamlit 默认 8501。
- 前端必须通过 `MNEMO_API_BASE_URL` 指向实际后端端口。

### Milvus 连接失败

检查 `.env` 中 `Milvus_url`、`Token`、`knowledge_base_collection`、`memory_collection` 是否与目标实例一致。应用启动阶段会校验 Milvus 连接，失败会直接终止启动。

### 重置本地数据

```powershell
docker compose down -v
```

这会删除 PostgreSQL 和 Redis 的本地数据卷；不会删除 Milvus/Zilliz 上的远程数据。
