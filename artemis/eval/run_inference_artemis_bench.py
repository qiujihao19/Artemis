import math
import os
import argparse
import json

import torch
import transformers
from tqdm import tqdm
import sys

from artemis.conversation import conv_templates, SeparatorStyle
from artemis.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, BOX_TOKEN_INDEX, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from artemis.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from artemis.model.builder import load_pretrained_model
from artemis.model.language_model.llava_llama import LlavaLlamaForCausalLM
from artemis.train.train import smart_tokenizer_and_embedding_resize
import pickle
import random
from PIL import Image
def avg(len_list,num_box):
    n = len(len_list) 
    interval = n // num_box
    samples = [len_list[i * interval] for i in range(num_box)]
    return samples

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--hc_stvg_val_file', help='Path to the hc-stvg val file.', required=True)
    parser.add_argument('--choose_mode', help='Frame selection strategy', default='average')
    parser.add_argument('--num_trackbox', help="The choosen trackbox number. If you do not use k-means, set the num_trackbox=num_inputbox", type=int, default=8)
    parser.add_argument('--num_inputbox', help="The extra box number input to LLM, if you use K-means, it should be equal to the K-means n-cluster", type=int, default=4)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def get_model_output(model, image_processor, tokenizer, video, sample, args):
    q = "What is the <bbox> doing in this video?"
    track_note = "This is the region's video tracking list: "
    if args.num_inputbox > 0:
        qs = q + ' ' + track_note + '<bbox>' * args.num_inputbox
    else:
        qs = q
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*1) + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*1) + '\n' + qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    video_tensor = video.half().to(args.device)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, BOX_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    
    tracklist = sample['track_list']
    frame_path = sample['frame_path']
    frame_id = sample['frame_id']
    no_fai_bbox_list = [ele for ele in range(len(tracklist)) if tracklist[ele] != []]
 
    if len(no_fai_bbox_list) > args.num_trackbox:
        if args.choose_mode == 'average':
            img_idx_list = avg(no_fai_bbox_list, args.num_trackbox)
        else:
            img_idx_list = random.sample(no_fai_bbox_list, args.num_trackbox)
    else:
        img_idx_list = no_fai_bbox_list
        for j in range((args.num_trackbox - len(img_idx_list))):
            img_idx_list.append(img_idx_list[-1])
    img_idx_list.sort()

    img_list = []
    box_list = []
    # choose the first image and box
    if sample['key_frame_box'] and sample['key_frame_id']:
        box_list.append(sample['key_frame_box'])
        file = sample['key_frame_id']
        img_list.append(Image.open(os.path.join(args.video_dir, frame_path, file)).convert('RGB'))
    else:
        input_id = random.choice(no_fai_bbox_list)
        img_idx_list = [input_id] + img_idx_list  
                            
    for cnt in img_idx_list:
        img_name = frame_id[cnt]
        img_path = os.path.join(args.video_dir, frame_path, img_name)
        img_list.append(Image.open(img_path).convert('RGB'))
        box_list.append(tracklist[cnt])
            
    assert len(img_list) == len(box_list) == args.num_trackbox + 1
    images_clip = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0].to(args.device) for i in img_list]
    
    w, h = img_list[0].size
    bboxes = [(torch.tensor(box) / torch.tensor([w, h, w, h], dtype=torch.half)).to((args.device)) for box in box_list]

    # ==========================================================================================================
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            images4box=images_clip,
            bboxes=bboxes,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)
    hc_stvg_file = json.load(open(args.hc_stvg_val_file, "r"))
    hc_stvg_file = get_chunk(hc_stvg_file, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results


    # Iterate over each sample in the ground truth file
    for sample in tqdm(hc_stvg_file):
        video_name = sample['video_name']
        groundtruth = sample['caption']

        sample_set = {'video_name': video_name, 'groundtruth': groundtruth}
        video_feature_path = os.path.join(args.video_dir, sample['clip_path'])
        if os.path.exists(video_feature_path):
            with open(video_feature_path, 'rb') as f:
                video_clip = pickle.load(f)
            video_clip = torch.tensor(video_clip)
            try:
                output = get_model_output(model, processor['image'], tokenizer, video_clip, sample, args)
                sample_set['pred'] = output
                output_list.append(sample_set)
            except Exception as e:
                print(f"Error processing video file '{video_name}': {e}")
            ans_file.write(json.dumps(sample_set) + "\n")        

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
