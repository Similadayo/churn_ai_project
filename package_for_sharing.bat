@echo off
REM ============================================================
REM  Creates a clean ZIP to send to someone else.
REM  It EXCLUDES venv, .git, caches and old zips so the
REM  recipient gets a small, portable copy that actually runs.
REM  Output: churn_ai_project_share.zip (next to this folder)
REM ============================================================
cd /d "%~dp0"

set "OUT=%~dp0..\churn_ai_project_share.zip"
if exist "%OUT%" del "%OUT%"

echo Building clean share zip...
powershell -NoProfile -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$exclude = @('venv','.git','__pycache__','.vscode');" ^
  "$items = Get-ChildItem -Force | Where-Object { $exclude -notcontains $_.Name -and $_.Extension -ne '.zip' };" ^
  "Compress-Archive -Path $items.FullName -DestinationPath '%OUT%' -Force"

if errorlevel 1 (
    echo [ERROR] Could not create the zip.
    pause
    exit /b 1
)

echo.
echo Done. Clean zip created at:
echo    %OUT%
echo Send THAT file. The recipient just unzips and double-clicks run.bat
echo.
pause
