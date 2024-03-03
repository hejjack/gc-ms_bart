# RASSP pretrain from scratch
CUDA_VISIBLE_DEVICES=1 python ../train_bart.py --config-file ../configs/train_config_pretrain_rassp1_neims1.yaml \
                                               --additional-info "_rassp1_neims1" \
                                               --additional-tags "A100:scratch:RASSP1NEIMS1:alfa" \
                                               --wandb-group pretrain \
                                               --checkpoint ../checkpoints/pretrain/bright-dumpling-376_rassp1_neims1/checkpoint-224000 \
                                               --resume-id w6pwi5mr


# ../checkpoints/bart_2023-04-07-18_27_23_30Mneims/checkpoint-1680000  # 30M_best
