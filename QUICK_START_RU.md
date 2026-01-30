# Quick Start - Supervised U-Net Segmentation (RU)

## 1) Training
```bash
python src/supervised_train.py --config configs/custom_config.yaml
```

## 2) Evaluation (val threshold + visuals)
```bash
python src/supervised_eval.py --config configs/custom_config.yaml --checkpoint experiments/my_custom_experiment/checkpoints/best_unet.pt
```

## 3) Production inference on large GeoTIFF
```bash
python src/production_unet_infer.py --image "path/to/large.tif" --checkpoint "experiments/my_custom_experiment/checkpoints/best_unet.pt" --config configs/custom_config.yaml --output-mask "experiments/production/mask.tif" --output-vector "experiments/production/mask.gpkg"
```

## Config hints
- normalization: per-band percentile clip (p_low/p_high)
- supervised.bands: all or list (e.g. [1,3,5])
- production.bands: all or list (e.g. "5,3,7")
- production.tile_size, production.context: speed/quality tradeoff
- production.batch_size: GPU acceleration
- production.vectorize: enable/disable vectorization
- ablations.enabled: run multi-band ablation sweep
