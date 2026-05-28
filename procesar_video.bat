@echo off
REM ==========================================================
REM  Procesador de video EPP YOLO + Verificador
REM  USO: arrastrá un archivo .mp4 hasta este .bat y soltalo.
REM       O ejecutalo y te pide el path manualmente.
REM ==========================================================

setlocal enabledelayedexpansion

REM Cambiar al directorio del .bat (raíz del proyecto)
cd /d "%~dp0"

echo.
echo ===================================================
echo   EPP YOLO + Verificador determinista
echo   Proyecto: Parcial Deep Learning
echo ===================================================
echo.

REM Verificar que el modelo existe
if not exist "runs_outputs\exp_D\best.pt" (
    echo [ERROR] No se encontro el modelo en:
    echo   runs_outputs\exp_D\best.pt
    echo.
    echo Descargalo del Release de GitHub:
    echo   https://github.com/PercyTechX/epp-yolo-compliance/releases/tag/v1.0
    echo.
    pause
    exit /b 1
)

REM Obtener el video a procesar
set "VIDEO_INPUT=%~1"

if "%VIDEO_INPUT%"=="" (
    echo No arrastraste ningun video.
    echo.
    set /p "VIDEO_INPUT=Pegame la ruta completa del video aqui: "
    REM Quitar comillas si las puso
    set "VIDEO_INPUT=!VIDEO_INPUT:"=!"
)

if not exist "%VIDEO_INPUT%" (
    echo.
    echo [ERROR] No se encontro el video: %VIDEO_INPUT%
    echo.
    pause
    exit /b 1
)

REM Construir el nombre del archivo de salida
for %%F in ("%VIDEO_INPUT%") do (
    set "VIDEO_NAME=%%~nF"
    set "VIDEO_EXT=%%~xF"
)

REM Crear carpeta de salida si no existe
if not exist "demo_video" mkdir "demo_video"

set "VIDEO_OUTPUT=demo_video\!VIDEO_NAME!_annotated.mp4"

echo.
echo --- Configuracion ---
echo Modelo:    runs_outputs\exp_D\best.pt
echo Entrada:   %VIDEO_INPUT%
echo Salida:    !VIDEO_OUTPUT!
echo Conf:      0.15
echo Imgsz:     1280
echo ---------------------
echo.

REM Ejecutar el pipeline
python -m src.run_video ^
    --weights "runs_outputs\exp_D\best.pt" ^
    --video "%VIDEO_INPUT%" ^
    --out "!VIDEO_OUTPUT!" ^
    --conf 0.15 ^
    --imgsz 1280

if errorlevel 1 (
    echo.
    echo [ERROR] El procesamiento fallo. Mira el mensaje arriba.
    echo.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo   LISTO
echo ===================================================
echo.
echo Video procesado guardado en:
echo   !VIDEO_OUTPUT!
echo.
echo Para verlo, abrilo con VLC u otro reproductor.
echo.
pause
