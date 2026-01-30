# Crop Segmentation Pipeline (Supervised U-Net)

This repository provides a supervised U-Net pipeline for crop field segmentation from satellite imagery.

## What works
- Supervised training/evaluation with U-Net
- Threshold selection on validation set
- TTA for evaluation (optional)
- Production inference on huge rasters with streaming tiles
- Vectorization of the predicted mask
- Flexible band selection (all bands or explicit channel lists)
- Percentile normalization (1?99) per band

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
- `normalization`: per-band percentile clip (p_low/p_high)
- `supervised.bands`: `all` or list (e.g. `[1,3,5]`)
- `production.bands`: `all` or list (e.g. `"5,3,7"`)
- `production.tile_size`, `production.context`: tile streaming settings
- `production.batch_size`: GPU batch size (when `use_tta: false`)
- `production.vectorize`: enable/disable vector output
- `ablations.enabled`: run multi-band ablation sweep

## Outputs
- `experiments/my_custom_experiment/checkpoints/best_unet.pt`
- `experiments/my_custom_experiment/supervised_eval/supervised_predictions_visualization.png`
- `experiments/production/midvolga_mask.tif`
- `experiments/production/midvolga_mask.gpkg`
