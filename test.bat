@echo off
pushd "%~dp0"
node --test tests/arch.test.js tests/diagram.test.js tests/inset_attention.test.js tests/inset_moe.test.js tests/inset_mixer.test.js
set EXITCODE=%ERRORLEVEL%
popd
exit /b %EXITCODE%
