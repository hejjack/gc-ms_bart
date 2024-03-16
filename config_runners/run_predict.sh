CUDA_VISIBLE_DEVICES=2 python ../predict.py --checkpoint ../checkpoints/finetune/resilient-lion-393_rassp1_neims1_224kPretrain/checkpoint-73738 \
                                            --output-folder ../predictions \
                                            --config-file ../configs/predict_config_nist_valid.yaml \
                                            # --data-range "0:200"