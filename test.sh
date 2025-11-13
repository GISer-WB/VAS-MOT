python main.py \
        --mode eval \
        --use-wandb False \
        --config-path ./configs/gmodet_r50_motip_vasmot.yaml \
        --inference-config-path ./configs/gmodet_r50_motip_vasmot.yaml \
        --data-root ./datasets4MOTIP/ \
        --outputs-dir ./outputs/gmodet_r50_motip_vasmot/ \
        --inference-model ./outputs/gmodet_r50_motip_vasmot/checkpoint_xx.pth