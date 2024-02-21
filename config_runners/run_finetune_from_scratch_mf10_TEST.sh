CUDA_VISIBLE_DEVICES=1 python ../train_bart.py --config-file ../configs/train_config_finetune_mf10_TEST.yaml \
                                               --additional-info _from_scratch_mf10_TEST \
                                               --wandb-group finetune \
                                               --additional-tags "A100:alfa:mf10"