
JSON_FOLDER="/path/to/train_json"
IMAGE_FOLDER="/path/to/LLaVA-Pretrain/images"
VIDEO_FOLDER="/path/to/valley/"

cd /path/to/Artemis
export PYTHONPATH="$/path/to/Artemis:${PYTHONPATH}"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed artemis/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /path/to/vicuna-v1.5-7b/ \
    --version v1 \
    --data_path ${JSON_FOLDER}/llava_image_.json ${JSON_FOLDER}/valley_processed_.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower /path/to/clip-vit-large-patch14 \
    --video_folder ${VIDEO_FOLDER} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/Artemis-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
