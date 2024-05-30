import os
import math
import json
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import CLIPVisionModel, CLIPImageProcessor
import cv2

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_video(vis_path, num_frm=100):
    image_list = os.listdir(vis_path)
    image_style = ['.jpg', '.png', 'JPEG']
    image_list = [file for file in image_list if file[-4:] in image_style]
    image_list = sorted(image_list, key = lambda x : int(x.split('.')))
    total_frame_num = len(image_list)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    image_array_list = []
    for idx in frame_idx:
        image_name = image_list[idx]
        image_array = cv2.imread(os.path.join(vis_path, image_name))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_array_list.append(image_array)
    img_array = np.stack(image_array_list, axis=0)
    a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def get_spatio_temporal_features(features, num_temporal_tokens=100):
    t, s, c = features.shape

    temporal_tokens = np.mean(features, axis=1)
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

    spatial_tokens = np.mean(features, axis=0)
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

    return sp_features


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--video_dir_path", default="", help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path", default="", help="The output dir to save the features in.")
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14", help="The path of vision tower")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--infer_batch", required=False, type=int, default=32,
                        help="Number of frames/images to perform batch inference.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    infer_batch = args.infer_batch
    os.makedirs(clip_feat_path, exist_ok=True)

    # Initialize the CLIP model
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower, torch_dtype=torch.float16)
    vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).cuda()
    vision_tower.eval()
    all_videos = []
    all_videos = os.listdir(video_dir_path)
    all_videos = get_chunk(all_videos, args.num_chunks, args.chunk_idx)
    video_clip_features = {}
    counter = 0
    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name
        if os.path.exists(f"{clip_feat_path}/{video_id}.pkl"):  # Check if the file is already processed
            continue
        try:
            video = load_video(video_path)
            video_tensor = image_processor.preprocess(video, return_tensors='pt')['pixel_values']
            video_tensor = video_tensor.half()

            n_chunk = len(video_tensor)
            video_features = torch.FloatTensor(n_chunk, 256, 1024).fill_(0)
            n_iter = int(math.ceil(n_chunk / float(infer_batch)))
            for i in range(n_iter):
                min_ind = i * infer_batch
                max_ind = (i + 1) * infer_batch
                video_batch = video_tensor[min_ind:max_ind].cuda()

                image_forward_outs = vision_tower(video_batch, output_hidden_states=True)

                select_hidden_state_layer = -2
                select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                batch_features = select_hidden_state[:, 1:]
                video_features[min_ind:max_ind] = batch_features.detach().cpu()

            video_clip_features[video_id] = get_spatio_temporal_features(video_features.numpy().astype("float16"))
            counter += 1

        except Exception as e:
            print(f"Can't process {video_path}")

        if counter % 512 == 0:  # Save after every 512 videos, update this number as per your requirements
            for key in video_clip_features.keys():
                features = video_clip_features[key]
                with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
                    pickle.dump(features, f)
            video_clip_features = {}

    for key in video_clip_features.keys():
        features = video_clip_features[key]
        with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
            pickle.dump(features, f)


if __name__ == "__main__":
    main()
