# finetune model ran on 4.8M rassp split generated by NEIMS dataset
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_finetune_NClike.yaml \
                                               --checkpoint ../checkpoints/pretrain/avid-aardvark-415_NClike/checkpoint-224000 \
                                               --additional-info "_NClike" \
                                               --wandb-group finetune  \
                                               --additional-tags "alfa:A100:NClike" \