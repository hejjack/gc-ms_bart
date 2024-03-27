# RASSP pretrain from scratch
python ../train_bart.py --config-file ../configs/train_config_pretrain_NClike.yaml \
                        --additional-info "_NClike" \
                        --additional-tags "A100:scratch:RASSP1NEIMS1:meta" \
                        --wandb-group pretrain \