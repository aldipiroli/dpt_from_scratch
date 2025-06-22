# Vision Transformers for Dense Prediction (DPT) from scratch
Implementing from scratch the paper "[Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413)" ICCV 2021, with the focus on the task of monocular depth estimation and semantic segmentation.

### Clone and install dependencies
``` 
git clone https://github.com/aldipiroli/dpt_from_scratch
pip install -r requirements.txt
``` 
### Train 
Depth Estimation on the [NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html):
- From pretrained [ViT16b](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html)
    ``` 
    cd python 
    python train.py config/dpt_depth_vit16b_config.yaml
    ```
- From scratch
    ``` 
    cd python 
    python train.py config/dpt_depth_config.yaml 
    ```


Semantic Segmentation on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/):
``` 
cd python 
python train.py config/dpt_semseg_config.yaml
```

>Note: the code implementation tries to follow the paper closely, though it occasionally differs from the [official repository](https://github.com/isl-org/DPT) in minor ways.