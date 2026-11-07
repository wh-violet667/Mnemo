@echo off
setlocal
set "PS1=%~dp001-enable-wsl.ps1"
echo Requesting administrator privileges...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath powershell -Verb RunAs -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','%PS1%')"
endlocal
