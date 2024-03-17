python ../predict.py --checkpoint ../checkpoints/finetune/icy-monkey-402_rassp1_neims1_336kPretrain/checkpoint-147476 \
                                            --output-folder ../predictions \
                                            --config-file ../configs/predict_config_nist_valid.yaml \
                                            # --data-range "0:200"