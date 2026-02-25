### DiT Linear Probing (MAE-style)

To run MAE-style linear probing on a pretrained **DiT-XL-2-256x256.pt** using 8 GPUs:

```bash
torchrun --nproc_per_node=8 linear_mae_style.py \
  --batch_size 512 \
  --epochs 90 \
  --lr 1.6e-3 \
  --warmup_epochs 10 \
  --weight_decay 0 \
  --data_path /path/to/imagenet \
  --output_dir /output/dir \
  --timestep 10 \
  --blockname layer-13
```

### Linear Probing (DDAE-style)

To run DDAE-style linear probing on a pretrained **DiT-XL-2-256x256.pt** using 8 GPUs:

```bash
torchrun --nproc_per_node=8 linear_ddae_style.py \
  --dataset in1k \
  --data_path /path/to/imagenet \
  --timestep 10 \
  --block_num layer-13 \
  --use_amp
```

Note: Added two linear probing pipelines to the original DiT repo: **`linear_mae_style.py`** (with MAE’s `util/` folder) and **`linear_ddae_style.py`** (with `utils.py`, `download.py`, and `datasets.py`). Updated `models.py`, following [DDAE](https://github.com/FutureXiang/ddae), to extract features from different DiT layers.
