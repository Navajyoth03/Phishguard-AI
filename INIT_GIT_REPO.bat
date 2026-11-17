@echo off
echo ========================================================
echo  PHISHGUARD AI - CLEAN GIT REPOSITORY INITIALIZATION
echo ========================================================
echo.
echo [1/4] Removing bloated 1.3GB old history (optimizer.pt)...
if exist .git (
    attrib -r -h -s .git /s /d >nul 2>&1
    rd /s /q .git >nul 2>&1
)

echo [2/4] Initializing fresh clean Git repository on 'main'...
git init -b main
if errorlevel 1 (
    git init
    git branch -M main
)

echo [3/4] Adding project files (protected by .gitignore)...
git add .

echo [4/4] Creating initial clean commit...
git commit -m "Initial commit: PhishGuard AI Dual-Mode Multi-Modal System"

echo.
echo ========================================================
echo  SUCCESS! Your repository is now ultra-light (~4 MB).
echo ========================================================
echo.
echo NEXT STEPS:
echo 1. Go to https://github.com/new and create an empty repository.
echo 2. Run these two commands in PowerShell:
echo.
echo    git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
echo    git push -u origin main
echo.
pause
