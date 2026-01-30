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
- supervised.bands: list of channels (e.g. 1..9)
- production.bands: channels for production inference
- production.tile_size, production.context: speed/quality tradeoff
- production.batch_size: GPU acceleration
- production.vectorize: enable/disable vectorization
