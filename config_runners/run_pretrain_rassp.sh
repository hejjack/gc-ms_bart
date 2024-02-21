CUDA_VISIBLE_DEVICES=1 python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp.yaml \
                                               --additional-info "_rassp_scratch" \
                                               --additional-tags "A100:RASSP:scratch:alfa" \
                                               --wandb-group pretrain \
