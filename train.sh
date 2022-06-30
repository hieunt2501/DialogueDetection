CUDA_VISIBLE_DEVICES="0" \
python ./scripts/train.py \
    --model_name_or_path keepitreal/vietnamese-sbert \
    --train_data_file ./data/films/train_bioes.csv \
    --eval_data_file ./data/films/eval_bioes.csv \
    --do_train \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_cycles 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-5 \
    --num_train_epochs 50 \
    --output_dir ./checkpoint/multi_task/v8 \
    --overwrite_output_dir \
    --adam_beta2 0.98 \
    --warmup_ratio 0.1 \
    --task dialogue \
    --fuse_lstm_information False \
    --residual False \
    --mask_speaker False \
    --speaker_classes 3 \
    --iob_classes 5 \
    --crf True \
    --top_k 1 \
    --sbert True \
    --fp16 False \
    --dataloader_pin_memory False \
    --dataloader_num_workers 0 \
    --use_device cpu