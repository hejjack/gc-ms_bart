
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_pretrain_neims.yaml \
                                               --additional-info "_neims_scratch" \
                                               --additional-tags "A100:NEIMS:scratch:alfa" \
                                               --wandb-group pretrain \

