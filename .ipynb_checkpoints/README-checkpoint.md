Clone of VLSAT repository to work with different others dataset.

0. Follow instruction for installation of original VLSAT: https://github.com/wz7in/CVPR2023-VLSAT.git
   
1. Generate data with ScanNet dataset. Go to data_processing/:

For GT ScanNet segmentation:

```bash
python gen_data_scannet_gt.py
```

For Mask3D ScanNet segmentation (use Chat-Scene segmentation files):

```bash
python gen_data_scannet_mask3d.py
```

For OneFormer3D ScanNet segmentation:

```bash
python gen_data_scannet_oneformer3d.py
```

2. Run patched evaluation with pre-trained VLSAT weights 3dssg_best_ckpt (available on https://github.com/wz7in/CVPR2023-VLSAT.git)

```bash
python -m main --mode eval --config config/mmgnet.json --exp /home/jovyan/Tatiana_Z/3dssg_best_ckpt
```

3. Run prepare_filtered_gnn_data.py (choose graph hyperparameters at the beginning of the script)

```bash
python prepare_filtered_gnn_data.py 
```