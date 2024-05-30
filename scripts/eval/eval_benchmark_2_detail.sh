
CKPT_NAME="Artemis-7b-lora"
Video_5_Benchmark="/path/to/video_chatgpt_bench"
pred_path="${Video_5_Benchmark}/${CKPT_NAME}/correctness_qa.json"
output_dir="${Video_5_Benchmark}/${CKPT_NAME}/gpt3/detail"
output_json="${Video_5_Benchmark}/${CKPT_NAME}/results/detail_qa.json"
api_key=""
api_base=""
num_tasks=8


python3 artemis/eval/eval_benchmark_2_detailed_orientation.py \
    --pred_path  ${pred_path} \
    --output_dir  ${output_dir} \
    --output_json  ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}
