@echo off
set VSLANG=1033
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
cmake -S "%~dp0." -B "%~dp0build" -G Ninja
if errorlevel 1 exit /b %errorlevel%
cmake --build "%~dp0build" --clean-first
if errorlevel 1 exit /b %errorlevel%
ctest --test-dir "%~dp0build" --verbose
exit /b %errorlevel%
