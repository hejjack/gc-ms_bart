CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_finetune.yaml \
                                               --additional-info _from_scratch \
                                               --wandb-group finetune \
                                               --additional-tags "A100:apollo"