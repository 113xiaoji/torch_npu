@echo off
setlocal
set VSLANG=1033
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
if not exist "%TEMP%\torch-npu-workspace-r41" mkdir "%TEMP%\torch-npu-workspace-r41"
cl /nologo /std:c++17 /EHsc /W4 /WX /I "%~dp0..\..\patches\linear_scaling_r41\op-plugin\op_plugin\utils" "%~dp0workspace_cache_test.cpp" /Fo"%TEMP%\torch-npu-workspace-r41\workspace_cache_test.obj" /Fe"%TEMP%\torch-npu-workspace-r41\workspace_cache_test.exe"
if errorlevel 1 exit /b %errorlevel%
"%TEMP%\torch-npu-workspace-r41\workspace_cache_test.exe"
exit /b %errorlevel%
