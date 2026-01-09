#!/bin/bash
# Quick start script for RL Crop Segmentation

set -e

echo "========================================="
echo "RL Crop Segmentation - Quick Start"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Check for data
echo ""
if [ ! -f "data/raw/satellite_image.tif" ] || [ ! -f "data/raw/crop_boundaries.shp" ]; then
    echo "⚠ WARNING: No data found in data/raw/"
    echo "Please place your GeoTIFF and Shapefile in data/raw/ directory:"
    echo "  - data/raw/satellite_image.tif"
    echo "  - data/raw/crop_boundaries.shp"
    echo ""
    echo "You can still run the setup, but training will fail without data."
    echo ""
else
    echo "✓ Data files found"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p experiments/{default,fast_training,high_quality}
mkdir -p data/{raw,processed}
mkdir -p notebooks
echo "✓ Directories created"

# Print next steps
echo ""
echo "========================================="
echo "Setup complete! 🎉"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Place your data in data/raw/:"
echo "   - satellite_image.tif (GeoTIFF)"
echo "   - crop_boundaries.shp (Shapefile)"
echo ""
echo "2. Run training:"
echo "   python src/train.py --config configs/fast_training.yaml"
echo ""
echo "3. Or use custom paths:"
echo "   python src/train.py \\"
echo "     --image data/raw/your_image.tif \\"
echo "     --shapefile data/raw/your_boundaries.shp \\"
echo "     --output experiments/my_experiment"
echo ""
echo "4. Evaluate trained model:"
echo "   python src/evaluate.py \\"
echo "     --checkpoint experiments/my_experiment/checkpoints/best_model.pt \\"
echo "     --config configs/default_config.yaml"
echo ""
echo "5. Run inference on new image:"
echo "   python src/inference.py \\"
echo "     --model experiments/my_experiment/checkpoints/best_model.pt \\"
echo "     --image data/raw/new_image.tif \\"
echo "     --output predictions/prediction.tif"
echo ""
echo "For more information, see README.md"
echo "========================================="
