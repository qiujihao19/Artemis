video_dir_path="/path/to/video_dir"
clip_feat_path="/path/to/save/clip_feat"
vision_tower="openai/clip-vit-large-patch14"
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 save_spatio_temporal_clip_features_video.py \
      --video_dir_path $video_dir_path \
      --clip_feat_path $clip_feat_path \
      --vision_tower $vision_tower \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done
