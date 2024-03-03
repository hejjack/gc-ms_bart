CUDA_VISIBLE_DEVICES=1 python ../train_bart.py --config-file ../configs/train_config_debug.yaml \
                                               --additional-info _ \
                                               --wandb-group debug     # finetune
                                               # --checkpoints_dir ../checkpoints \
                                               # --resume-id 2yqhwcas