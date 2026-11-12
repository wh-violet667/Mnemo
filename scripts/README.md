# 在 D 盘安装 Docker（方案 A）

安装包已经下载到：

```
D:\Docker\installer\Docker Desktop Installer.exe  (约 595 MB)
```

这台机器的情况：

- Windows 11 家庭版，虚拟化已开启（HypervisorPresent=True）
- D 盘剩余 31.6 GB，足够安装
- WSL / VirtualMachinePlatform 目前是**关闭**状态，需要管理员权限开启并重启一次
- 当前 shell 不是管理员，所以下面第 1 步必须由你右键“以管理员身份运行”

## 1. 开启 WSL（管理员）

> **不要直接运行 Docker 安装包**：per-user 模式没有权限开启 VirtualMachinePlatform，会报你看到的那个错误。

最简单的方式：双击

```
scripts\01-enable-wsl.cmd
```

它会弹出 UAC，点“是”；窗口跑完 DISM 后按回车即可。

如果双击没反应，再手动执行（右键 PowerShell → “以管理员身份运行”）：

```powershell
cd "C:\Users\27843\Desktop\ai agent\Mnemo-master"
powershell -ExecutionPolicy Bypass -File .\scripts\01-enable-wsl.ps1
```

也可以从普通 PowerShell 直接弹 UAC：

```powershell
Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"$PWD\scripts\01-enable-wsl.ps1"
```

然后重启 Windows。

> 如果重启后 `02-install-docker.ps1` 仍提示 WSL 未就绪，请再以管理员身份运行一次
> `wsl --install --no-distribution` 或 `wsl --update`，然后重启。

## 2. 安装 Docker Desktop 到 D 盘（普通用户）

重启后，在项目目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\02-install-docker.ps1
```

它会：

- 用 per-user 模式安装到 `D:\Docker\DockerDesktop`
- 把 WSL 磁盘放到 `D:\Docker\wsl`
- 启动 Docker Desktop 并等待引擎就绪

首次启动 Docker Desktop 可能需要登录 Docker 账号、接受协议；按界面提示完成即可。

## 3. 启动 PostgreSQL 和 Redis Stack

```powershell
docker compose up -d
docker compose ps
```

## 4. 配置项目

```powershell
copy .env.docker.example .env
```

然后填 `OPENCODE_API_KEY`、`DASHSCOPE_API_KEY`（embedding / rerank）、`Milvus_url`、`Token`。

## 5. 启动项目

后端（8000 被 CLodop 占用，用 8010）：

```powershell
cd backend
$env:PORT=8010
..\.venv\Scripts\python.exe main.py
```

前端：

```powershell
$env:MNEMO_API_BASE_URL="http://localhost:8010"
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py --server.port 8501
```

## 卸载/重置

```powershell
docker compose down -v   # 删除容器和数据卷
```
