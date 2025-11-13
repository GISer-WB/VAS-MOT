CUDA_VISIBLE_DEVICES=0 python main.py \
        --mode train \
        --use-wandb False \
        --config-path ./configs/gmodet_r50_motip_vasmot.yaml \
        --data-root ./datasets4MOTIP/ \
        --outputs-dir ./outputs/gmodet_r50_motip_vasmot/
