@echo off
chcp 65001 >nul
title Video Test Runner - Batch Mode

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║         VIDEO TEST RUNNER - BATCH MODE                   ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM Default paths
set "CONFIG_FILE=test_config.json"
set "TEST_DIR=F:\test"
set "SINGLE_VIDEO="

REM Menu
echo [1] Test tek video (F:\test\test.mp4)
echo [2] Test tüm videolar (F:\test klasöründeki tüm .mp4)
echo [3] Custom video seç
echo [4] Custom klasör seç
echo.
set /p choice="Seçim (1-4): "

if "%choice%"=="1" (
    set "SINGLE_VIDEO=%TEST_DIR%\test.mp4"
    goto :run_single
)

if "%choice%"=="2" (
    goto :run_folder
)

if "%choice%"=="3" (
    set /p SINGLE_VIDEO="Video dosyası tam yolu: "
    goto :run_single
)

if "%choice%"=="4" (
    set /p TEST_DIR="Klasör yolu: "
    goto :run_folder
)

echo Geçersiz seçim!
pause
exit /b 1

:run_single
echo.
echo ═══════════════════════════════════════════════════════════
echo  Tek video test başlatılıyor...
echo  Video: %SINGLE_VIDEO%
echo  Config: %CONFIG_FILE%
echo ═══════════════════════════════════════════════════════════
echo.

if not exist "%SINGLE_VIDEO%" (
    echo HATA: Video bulunamadı: %SINGLE_VIDEO%
    pause
    exit /b 1
)

python test_runner.py --config "%CONFIG_FILE%" "%SINGLE_VIDEO%"
goto :end

:run_folder
echo.
echo ═══════════════════════════════════════════════════════════
echo  Klasör test başlatılıyor...
echo  Klasör: %TEST_DIR%
echo  Config: %CONFIG_FILE%
echo  Pattern: *.mp4
echo ═══════════════════════════════════════════════════════════
echo.

if not exist "%TEST_DIR%" (
    echo HATA: Klasör bulunamadı: %TEST_DIR%
    pause
    exit /b 1
)

python test_runner.py --config "%CONFIG_FILE%" --folder "%TEST_DIR%" --pattern "*.mp4"
goto :end

:end
echo.
echo ═══════════════════════════════════════════════════════════
echo  Test tamamlandı!
echo ═══════════════════════════════════════════════════════════
echo.
pause
