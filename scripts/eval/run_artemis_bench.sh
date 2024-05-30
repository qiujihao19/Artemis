

model_path="checkpoints/${CKPT_NAME}"
model_base="checkpoints/Artemis-7b-ftune"
cache_dir="./cache_dir"
hc_stvg_val_file="/path/to/videoref_json/videorefbench_json/hc_stvg_val.json"
video_dir="/path/to/artemis_dataset"
output_dir="${video_dir}/${CKPT_NAME}"
choose_mode="average"
num_trackbox=8
num_inputbox=4


gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

export PYTHONPATH="/path/to/Artemis:${PYTHONPATH}"

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 artemis/eval/run_inference_artemis_bench.py \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --hc_stvg_val_file ${hc_stvg_val_file} \
      --choose_mode ${choose_mode} \
      --num_trackbox ${num_trackbox} \
      --num_inputbox ${num_inputbox} \
      --model_base ${model_base} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done