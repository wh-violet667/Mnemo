$ErrorActionPreference = "Stop"

$installer = "D:\Docker\installer\Docker Desktop Installer.exe"
$installDir = "D:\Docker\DockerDesktop"
$dataDir = "D:\Docker\wsl"

if (-not (Test-Path $installer)) {
    Write-Error "找不到安装包：$installer。请确认下载完成。"
    exit 1
}

Write-Host "== 检查 WSL ==" -ForegroundColor Cyan
$vmPlatform = Get-CimInstance Win32_OptionalFeature -Filter "Name='VirtualMachinePlatform'"
if (-not $vmPlatform -or $vmPlatform.InstallState -ne 1) {
    Write-Warning "VirtualMachinePlatform 尚未启用。"
    Write-Warning "请先双击运行 scripts\01-enable-wsl.cmd（会弹 UAC），重启后再运行本脚本。"
    Read-Host "按回车退出"
    exit 1
}

wsl --status *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "WSL 尚未就绪。请以管理员身份运行 wsl --install --no-distribution，重启后再试。"
    Read-Host "按回车退出"
    exit 1
}

Write-Host "== 安装 Docker Desktop 到 D 盘（当前用户，无需管理员）==" -ForegroundColor Cyan
Start-Process -FilePath $installer -Wait -ArgumentList @(
    "install",
    "--user",
    "--accept-license",
    "--backend=wsl-2",
    "--installation-dir=$installDir",
    "--wsl-default-data-root=$dataDir"
)

$dockerExe = Join-Path $installDir "Docker Desktop.exe"
if (-not (Test-Path $dockerExe)) {
    Write-Warning "安装完成，但没有在 $installDir 找到 Docker Desktop.exe。请检查安装日志。"
    Read-Host "按回车退出"
    exit 1
}

Write-Host "== 启动 Docker Desktop ==" -ForegroundColor Cyan
Start-Process $dockerExe

Write-Host "== 等待 Docker 引擎启动（最多 120 秒）==" -ForegroundColor Cyan
$dockerCli = Join-Path $installDir "resources\bin\docker.exe"
if (-not (Test-Path $dockerCli)) {
    $dockerCli = "docker"
}
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    & $dockerCli info *> $null
    if ($LASTEXITCODE -eq 0) {
        $ready = $true
        break
    }
    Start-Sleep -Seconds 2
}

if ($ready) {
    Write-Host "Docker 已就绪。" -ForegroundColor Green
    Write-Host "下一步：在项目根目录运行  docker compose up -d" -ForegroundColor Yellow
} else {
    Write-Warning "Docker 还没启动完成。请打开 Docker Desktop 手动确认（首次可能需要登录/接受协议）。"
}
Read-Host "按回车退出"
