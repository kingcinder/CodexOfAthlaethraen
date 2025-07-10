@echo off
set GITHUB_USERNAME=kingcinder
set GITHUB_TOKEN=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
set REPO_NAME=CodexOfAthlaethraen
set LOCAL_DIR=C:\CodexPush

cd /d %LOCAL_DIR%
git init
git remote add origin https://%GITHUB_USERNAME%:%GITHUB_TOKEN%@github.com/%GITHUB_USERNAME%/%REPO_NAME%.git
git add .
git commit -m "Initial push of Codex files – Volume I, Echo Log, Ritual Progress"
git branch -M main
git push -u origin main --force


echo.
echo Codex push complete. Verify at: https://github.com/%GITHUB_USERNAME%/%REPO_NAME%
pause

