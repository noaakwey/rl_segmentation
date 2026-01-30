# Crop Segmentation Pipeline (Supervised U-Net)

This repository provides a supervised U-Net pipeline for crop field segmentation from satellite imagery.

## What works
- Supervised training/evaluation with U-Net
- Threshold selection on validation set
- TTA for evaluation (optional)
- Production inference on huge rasters with streaming tiles
- Vectorization of the predicted mask

## Quick start

### 1) Train U-Net
```bash
python src/supervised_train.py --config configs/custom_config.yaml
```

### 2) Evaluate U-Net (val threshold + plots)
```bash
python src/supervised_eval.py --config configs/custom_config.yaml --checkpoint experiments/my_custom_experiment/checkpoints/best_unet.pt
```

### 3) Production inference on huge rasters
```bash
python src/production_unet_infer.py --image "path/to/large.tif" --checkpoint "experiments/my_custom_experiment/checkpoints/best_unet.pt" --config configs/custom_config.yaml --output-mask "experiments/production/mask.tif" --output-vector "experiments/production/mask.gpkg"
```

## Config highlights
- `supervised.bands`: which bands to use (supports 4, 5, 6, 7, 8, 9+ channels)
- `production.bands`: which bands to use during production inference
- `production.tile_size`, `production.context`: tile streaming settings
- `production.batch_size`: GPU batch size (when `use_tta: false`)
- `production.vectorize`: enable/disable vector output

## Outputs
- `experiments/my_custom_experiment/checkpoints/best_unet.pt`
- `experiments/my_custom_experiment/supervised_eval/supervised_predictions_visualization.png`
- `experiments/production/midvolga_mask.tif`
- `experiments/production/midvolga_mask.gpkg`
