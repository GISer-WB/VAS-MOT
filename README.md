# **Multi-object tracking of vehicles and anomalous states in remote sensing videos: Joint learning of historical trajectory guidance and ID prediction**

* The following code is for the paper titled "Multi-object tracking of vehicles and anomalous states in remote sensing videos: Joint learning of historical trajectory guidance and ID prediction".
* This paper was published in the journal "ISPRS Journal of Photogrammetry and Remote Sensing".

## Update
- [x] 🎉 Pre-trained weights will come soon.
- [x] 🍀  [2025/11/23] Released the VAS-MOT code；

## Installation

Currently, VAS-MOT is built upon ​**Python 3.11, PyTorch 2.2 (recommended)​**.

🛠️ Installation command

```bash
conda create -n VAS-MOT python=3.11		# suggest to use virtual envs
conda activate VAS-MOT
# PyTorch:
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia		# CUDA version=12.1 is also OK
# Other dependencies:
conda install matplotlib pyyaml scipy tqdm tensorboard seaborn scikit-learn pandas
pip install opencv-python einops wandb pycocotools timm
# Compile the Deformable Attention:
cd models/ops/
sh make.sh
# After compiled, you can use following script to test it:
python test.py		# [Optional]
```

⚠️ mamba-ssm can be cownload from [mamba](https://github.com/state-spaces/mamba/releases/tag/v2.2.4) and install via pip.

## Useage

1. train

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
        --mode train \
        --use-wandb False \
        --config-path ./configs/gmodet_r50_motip_vasmot.yaml \
        --data-root ./datasets4MOTIP/ \
        --outputs-dir ./outputs/gmodet_r50_motip_vasmot/
```

2. test

```
CUDA_VISIBLE_DEVICES=0 python main.py \
        --mode eval \
        --use-wandb False \
        --config-path ./configs/gmodet_r50_motip_vasmot.yaml \
        --inference-config-path ./configs/gmodet_r50_motip_vasmot.yaml \
        --data-root ./datasets4MOTIP/ \
        --outputs-dir ./outputs/gmodet_r50_motip_vasmot/ \
        --inference-model ./outputs/gmodet_r50_motip_vasmot/checkpoint_xx.pth
```

## Evalution

First, `cd TrackEval, then run `python scripts/run_mot_challenge.py --TRACKERS_TO_EVAL VAS-MOT

To evaluate your results,

* put your results in `TrackEval/data/trackers/mot_challenge/MOT15-val/VAS-MOT`,
* run `python scripts/run_mot_challenge.py --TRACKERS_TO_EVAL VAS-MOT`

The specific path and data format can be adjusted according to your own data format.

## Acknowledgements

This project is built upon [MOTIP](https://github.com/Annzstbl/MOTIP), [RT-DETR](https://github.com/lyuwenyu/RT-DETR), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [MOTR](https://github.com/megvii-research/MOTR), [DAB-Deformable DETR](https://github.com/IDEA-Research/DAB-DETR), [TrackEval](https://github.com/JonathonLuiten/TrackEval). Thanks to the contributors of these great codebases.
