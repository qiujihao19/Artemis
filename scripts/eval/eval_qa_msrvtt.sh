
GPT_Zero_Shot_QA="/path/to/MSRVTT"
output_name="Artemis-7b-lora"
pred_path="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${output_name}/merge.jsonl"
output_dir="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${output_name}/gpt"
output_json="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${output_name}/results.json"
api_key=""
api_base=""
num_tasks=2



python3 artemis/eval/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}
