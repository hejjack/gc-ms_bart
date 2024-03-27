# RASSP pretrain from scratch
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1_200peaks.yaml \
                    --additional-info "_rassp1_neims1_200peaks_masks" \
                    --additional-tags "A100:scratch:RASSP1NEIMS1:meta" \
                    --wandb-group pretrain \