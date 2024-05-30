
GPT_Zero_Shot_QA="/path/to/TGIF"
output_name="Artemis-7b-lora"
pred_path="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/${output_name}/merge.jsonl"
output_dir="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/${output_name}/gpt3.5-0.0"
output_json="${GPT_Zero_Shot_QA}/TGIF_Zero_Shot_QA/${output_name}/results.json"
api_key=""
api_base=""
num_tasks=8



python3 artemis/eval/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}
