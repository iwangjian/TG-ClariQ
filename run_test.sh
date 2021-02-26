CUDA_VISIBLE_DEVICES=0 python3 src/main.py --do_test \
        --data_dir=data \
        --log_dir=log/ours/bert \
        --output_dir=output/ours/bert \
        --text_encoder=bert \
        --bert_model_dir=pretrain/bert/base-uncased \
        --max_vocab_size=30000 \
        --batch_size=8
