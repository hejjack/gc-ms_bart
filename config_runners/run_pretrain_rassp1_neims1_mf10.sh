# RASSP1 NEIMS1 pretrain from scratch
CUDA_VISIBLE_DEVICES=0 python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1_mf10.yaml \
                                               --additional-info "_rassp1_neims1_mf10" \
                                               --additional-tags "A100:scratch:RASSP1NEIMS1:meta" \
                                               --wandb-group pretrain \
                                               --resume-id 9b1ifnhd \
                                               --checkpoint ../checkpoints/pretrain/desert-feather-378_rassp1_neims1_mf10/checkpoint-88000
                                        






