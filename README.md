# UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN) #
## Introduction
This repository contains a PyTorch implementation of a graph-based vision transformer (GraphTrans) framework for whole slide image (WSI) classification.
The data employed in this work can be found from the **UBC-OCEAN Kaggle challenge** https://www.kaggle.com/competitions/UBC-OCEAN/overview

## How to use

### Preprocessing
1. Download the UBC-OCEAN dataset and put it in _./data_ folder
2. Run the patch-tiling code using the following command
   ```
   python src/tile_WSI.py -s 512 -e 0 -j 32 -B 50 -M 20 -o ./data/output_tiled ./data/UBC-OCEAN/train_images/*.png
   ```
3. Contruct the graph using the following command
  ```
  python build_graphs.py --weights ./resnet18.pth --dataset ../data/output_tiled --output ../graphs
  ```

### Training GraphTrans
Run the following command to train and store the model and log files
  ```
  bash scripts/train.sh
  ```
To evaluate the model run ``bash script/test.sh``

**NOTE: You can change the training settings in the bash files.**

### GraphCAM
To generate the saliency maps of the model on the WSI, run the following command
  ```
  bash scripts/get_graphcam.sh
  bash scripts/vis_graphcam.sh
  ```
