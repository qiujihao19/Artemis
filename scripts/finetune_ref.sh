
IMAGE_FOLDER=""
VIDEO_FOLDER="/path/to/DATA_ROOT/Artemis_data"
DATA_FOLDER="/path/to/videoref_json"
cd /path/to/Artemis

export PYTHONPATH="path/:${PYTHONPATH}"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed artemis/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path ./checkpoints/Artemis-7b-ftune \
    --version v1 \
    --data_path ${DATA_FOLDER} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower path/to/clip-vit-large-patch14 \
    --video_folder ${VIDEO_FOLDER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir ./checkpoints/Artemis-7b-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 20 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"\
    --mm_use_bbox_token False\
    --spi_model True \
    --k_means True \
    --choosen_mode average \
    --train_mode referring\
    --lora_enable True \
    --lora_r 16
