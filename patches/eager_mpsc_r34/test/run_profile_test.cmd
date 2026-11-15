@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b 1
cmake -S "%~dp0." -B "%~dp0build" -G Ninja
if errorlevel 1 exit /b 1
cmake --build "%~dp0build" --clean-first
if errorlevel 1 exit /b 1
ctest --test-dir "%~dp0build" --output-on-failure
exit /b %errorlevel%
