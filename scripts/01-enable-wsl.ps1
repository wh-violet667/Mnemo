#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"

Write-Host "== 1/2 启用 WSL 与虚拟机平台（需要重启）==" -ForegroundColor Cyan
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

Write-Host "== 2/2 安装 WSL2（如果失败，重启后再运行本脚本）==" -ForegroundColor Cyan
wsl --install --no-distribution
if ($LASTEXITCODE -ne 0) {
    Write-Host "默认安装失败，改用 --web-download 重试..." -ForegroundColor Yellow
    wsl --install --no-distribution --web-download
}
if ($LASTEXITCODE -ne 0) {
    Write-Warning "WSL 仍未安装成功。请检查网络，或手动运行：wsl --install --no-distribution --web-download"
}

Write-Host ""
Write-Host "完成。请重启 Windows，然后运行：" -ForegroundColor Yellow
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\02-install-docker.ps1"
Write-Host ""
Read-Host "按回车退出"
