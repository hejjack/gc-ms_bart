# RASSP pretrain from scratch
python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1_200peaks.yaml \
                        --additional-info "_rassp1_neims1_200peaks" \
                        --additional-tags "A100:scratch:RASSP1NEIMS1:meta" \
                        --wandb-group pretrain \