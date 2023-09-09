CUDA_VISIBLE_DEVICES=2 python ../train_bart.py --config-file train_config_finetune.yaml \
                                               --checkpoint ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000 \
                                               --additional-info "" \
                                               --wandb-group finetune     # finetune
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \
                                               # --resume-id 2yqhwcas

# run on A40
CUDA_VISIBLE_DEVICES=2 python ../train_bart.py --config-file train_config_finetune.yaml \
                                               --additional-info "_from_scratch" \
                                               --wandb-group finetune     # finetune
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \
                                               # --resume-id 2yqhwcas