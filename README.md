# **Multi-object tracking of vehicles and anomalous states in remote sensing videos: Joint learning of historical trajectory guidance and ID prediction**

* The following code is for the paper titled "Multi-object tracking of vehicles and anomalous states in remote sensing videos: Joint learning of historical trajectory guidance and ID prediction".
* This paper is currently under review in the journal "ISPRS Journal of Photogrammetry and Remote Sensing".

## Update
- [x] 🎉 Pre-trained weights will come soon.
- [x] 🍀  [2025/11/13] Released the VAS-MOT code；

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

⚠️ mamba-ssm can be cownload from [mamba]([https://](https://github.com/state-spaces/mamba/releases/tag/v2.2.4)) and install via pip.
