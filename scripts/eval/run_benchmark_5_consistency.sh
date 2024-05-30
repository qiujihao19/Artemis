

CKPT_NAME="Artemis-7b-lora"
model_path="checkpoints/${CKPT_NAME}"
model_base="checkpoints/Artemis-7b-ftune"
cache_dir="./cache_dir"
Video_5_Benchmark="/path/to/video_chatgpt_bench"
video_dir="${Video_5_Benchmark}/video_clip_feature"
gt_file="${Video_5_Benchmark}/consistency_qa.json"
output_dir="${Video_5_Benchmark}/${CKPT_NAME}"
output_name="consistency_qa"

export PYTHONPATH="/path/to/Artemis:${PYTHONPATH}"
CUDA_VISIBLE_DEVICES=2 python3 artemis/eval/run_inference_benchmark_consistency.py \
    --model_path ${model_path} \
    --model_base ${model_base} \
    --cache_dir ${cache_dir} \
    --video_dir ${video_dir} \
    --gt_file ${gt_file} \
    --output_dir ${output_dir} \
    --output_name ${output_name}