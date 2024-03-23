python ../train_bart.py --config-file ../configs/train_config_finetune_mf10000000_200peaks.yaml \
                        --checkpoint ../checkpoints/pretrain/glowing-puddle-405_rassp1_neims1_200peaks/checkpoint-224000 \
                        --additional-info "_rassp1_neims1_200pt_200ft" \
                        --wandb-group finetune  \
                        --additional-tags "meta" \