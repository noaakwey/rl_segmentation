@echo off
echo =========================================
echo RL Crop Segmentation - Quick Start (Windows)
echo =========================================
echo.

:: Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    echo ^Check Virtual environment created
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
echo ^Check Virtual environment activated

:: Install dependencies
echo.
echo Installing dependencies...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
echo ^Check Dependencies installed

:: Check for data
echo.
if not exist "data\raw\satellite_image.tif" (
    echo ^! WARNING: satellite_image.tif not found in data\raw\
) else if not exist "data\raw\crop_boundaries.shp" (
    echo ^! WARNING: crop_boundaries.shp not found in data\raw\
) else (
    echo ^Check Data files found
)

if not exist "data\raw\satellite_image.tif" (
    echo Please place your GeoTIFF and Shapefile in data\raw\ directory:
    echo   - data\raw\satellite_image.tif
    echo   - data\raw\crop_boundaries.shp
    echo.
    echo You can still run the setup, but training will fail without data.
    echo.
)

:: Create necessary directories
echo Creating directories...
if not exist experiments\default mkdir experiments\default
if not exist experiments\fast_training mkdir experiments\fast_training
if not exist experiments\high_quality mkdir experiments\high_quality
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist notebooks mkdir notebooks
echo ^Check Directories created

:: Print next steps
echo.
echo =========================================
echo Setup complete! ^!^!
echo =========================================
echo.
echo Next steps:
echo.
echo 1. Place your data in data\raw\:
echo    - satellite_image.tif (GeoTIFF)
echo    - crop_boundaries.shp (Shapefile)
echo.
echo 2. Run training:
echo    python src/train.py --config configs/fast_training.yaml
echo.
echo 3. Or use custom paths:
echo    python src/train.py ^
     --image data/raw/your_image.tif ^
     --shapefile data/raw/your_boundaries.shp ^
     --output experiments/my_experiment
echo.
echo 4. Evaluate trained model:
echo    python src/evaluate.py ^
     --checkpoint experiments/my_experiment/checkpoints/best_model.pt ^
     --config configs/default_config.yaml
echo.
echo 5. Run inference on new image:
echo    python src/inference.py ^
     --model experiments/my_experiment/checkpoints/best_model.pt ^
     --image data/raw/new_image.tif ^
     --output predictions/prediction.tif
echo.
echo For more information, see README.md
echo =========================================
pause
