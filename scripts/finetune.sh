

JSON_FOLDER="/path/to/train_json"
IMAGE_FOLDER="/path/to/llava_finetune"
VIDEO_FOLDER="/path/to/video_clip_feature"

cd /path/to/Artemis

export PYTHONPATH="${HOME}/Workspace/video/Artemis:${PYTHONPATH}"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed artemis/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path /media/Disk1/Dataset/LLaVA/vicuna-v1.5-7b/ \
    --version v1 \
    --data_path ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/videochatgpt_tune_.json ${JSON_FOLDER}/nlp_tune.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower /path/to/clip-vit-large-patch14 \
    --video_folder ${VIDEO_FOLDER} \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/Artemis-7b-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/Artemis-7b-ftune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
