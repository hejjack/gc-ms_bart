# finetune model ran on 1-1-0.1 dataset (NEIMS1, RASSP1, NIST01)
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_finetune_mf10000000.yaml \
                                               --checkpoint ../checkpoints/pretrain/fortuitous-monkey-376_rassp1_neims1_neims01/checkpoint-112000 \
                                               --additional-info "_rassp1_neims1_nist01" \
                                               --wandb-group finetune  \
                                               --additional-tags "alfa:A100:1-1-0.1" \
                                            #    --device cpu \
                                               # --checkpoints_dir ../checkpoints \