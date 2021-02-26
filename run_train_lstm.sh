CUDA_VISIBLE_DEVICES=0 python3 src/main.py --do_train \
        --data_dir=data \
        --log_dir=log/ours/lstm \
        --text_encoder=lstm \
        --max_vocab_size=30000 \
        --embed_file=pretrain/glove/glove.42B.300d.txt \
        --embed_size=300 \
        --ns_num=2 \
        --hidden_size=256 \
        --num_attention_heads=8 \
        --num_layers=3 \
        --batch_size=16 \
        --num_epochs=10 \
        --log_steps=100 \
        --validate_steps=1000 \
        --lr=2e-4