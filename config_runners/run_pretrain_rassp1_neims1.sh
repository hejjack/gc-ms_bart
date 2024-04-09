# RASSP pretrain from scratch
python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1_224k.yaml \
                        --additional-info "_rassp1_neims1" \
                        --additional-tags "H100:scratch:rassp1:neims1:meta" \
                        --wandb-group pretrain \
                        --checkpoint ../checkpoints/pretrain/blooming-thunder-439_rassp1_neims1/checkpoint-112000 \
                        --resume-id "ai0xxvoh"