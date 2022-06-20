CUDA_VISIBLE_DEVICES="0" \
python ./scripts/train.py \
    --model_name_or_path vinai/phobert-base \
    --train_data_file ./data/speaker_diarization/train.csv \
    --eval_data_file ./data/speaker_diarization/eval.csv \
    --do_train \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 32 \
    --num_cycles 1 \
    --weight_decay 0.1 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --use_device cpu \
    --dataloader_pin_memory False \
    --dataloader_num_workers 0 \
    --output_dir ./checkpoint/multitask/v11_correction_bias \
    --overwrite_output_dir \
    --adam_beta2 0.98 \
    --warmup_ratio 0.1 \
    --task speaker