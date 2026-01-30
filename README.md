# Crop Segmentation Pipeline (Supervised U-Net)

This repository provides a supervised U-Net pipeline for crop field segmentation from satellite imagery.

## Status
- RL pipeline removed
- Supervised U-Net is the primary workflow

## Requirements
- Python 3.8+
- CUDA (optional, recommended for training/large inference)

Install deps:
```bash
pip install -r requirements.txt
```

## Data format
- GeoTIFF image (multi-band) in `image_path`
- Shapefile (polygons) in `shapefile_path`
- Bands are 1-based indices (RasterIO style)

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
python src/production_unet_infer.py \
  --image "path/to/large.tif" \
  --checkpoint "experiments/my_custom_experiment/checkpoints/best_unet.pt" \
  --config configs/custom_config.yaml \
  --output-mask "experiments/production/mask.tif" \
  --output-vector "experiments/production/mask.gpkg"
```

## Config details

### Core
- `image_path`, `shapefile_path`: input paths
- `patch_size`, `stride`: patch extraction
- `train_ratio`, `val_ratio`, `test_ratio`
- `output_dir`: experiment output folder

### Normalization
Per-band percentile clip by default (1-99). Useful for mixed data ranges.
```yaml
normalization:
  mode: percentile
  p_low: 1
  p_high: 99
```

### Band selection (universal)
You can use all bands or any custom list/combination.
```yaml
supervised:
  bands: all         # use all bands
# or
supervised:
  bands: [1, 3, 5]   # custom order
```
Production uses the same idea:
```yaml
production:
  bands: all
# or
production:
  bands: "5,3,7"
```

### Training (supervised)
```yaml
supervised:
  batch_size: 8
  num_epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  min_learning_rate: 0.000001
  lr_reduce_factor: 0.5
  lr_reduce_patience: 5
  early_stopping_patience: 10
  early_stopping_min_delta: 0.001
  bce_weight: 0.5
  dice_weight: 0.5
  balance: true
  min_crop_fraction: 0.01
  base_channels: 32
```

### Evaluation
- Best threshold is selected on val, then applied on test.
- Optional TTA (flip-based) during evaluation.
```yaml
supervised_eval:
  threshold_min: 0.3
  threshold_max: 0.7
  threshold_steps: 9
  n_visualize: 6
  use_tta: true
```

### Production inference
```yaml
production:
  tile_size: 1024
  context: 128
  threshold: 0.7
  use_tta: false
  batch_size: 8
  bands: all
  min_area_pixels: 4096
  vectorize: true
```
Notes:
- `use_tta: false` enables batched inference (faster)
- Increase `tile_size` if you have enough VRAM
- Set `vectorize: false` to skip vector output

## Ablations (optional)
Ablations run multiple experiments with different band sets. This helps you find the best combination.
By default, ablations are disabled.
```yaml
ablations:
  enabled: false
  run_eval: true
  variants:
    - name: rgb
      bands: [1, 2, 3]
    - name: rgb_nir
      bands: [1, 2, 3, 4]
    - name: indices_only
      bands: [5, 6, 7]
```
To run ablations, set `enabled: true` and re-run training.
Summary is saved to:
```
experiments/ablations/ablations_summary.csv
```

## Outputs
- `experiments/my_custom_experiment/checkpoints/best_unet.pt`
- `experiments/my_custom_experiment/supervised_eval/supervised_predictions_visualization.png`
- `experiments/production/midvolga_mask.tif`
- `experiments/production/midvolga_mask.gpkg`

## Testing
No automated tests are included yet.

## Troubleshooting
- If you see channel mismatch, check `supervised.bands` and `production.bands`.
- For size errors, ensure tile sizes are padded internally (already handled in production).
- For slow inference, reduce TTA and increase batch size if VRAM allows.
