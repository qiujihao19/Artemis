import os
import argparse
import json
from tqdm import tqdm

from artemis.eval.run_inference_video_qa import get_model_output
from artemis.mm_utils import get_model_name_from_path
from artemis.model.builder import load_pretrained_model
import pickle
import torch

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    # parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)


    return parser.parse_args()


def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)

    # Load the ground truth file
    with open(args.gt_file) as file:
        gt_contents = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    # conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_contents):
        video_name = sample['video_name']
        sample_set = sample
        question_1 = sample['Q1']
        question_2 = sample['Q2']

        video_clip_name = f"{video_name}{'.pkl'}"
        video_feature_path = os.path.join(args.video_dir, video_clip_name)
        if os.path.exists(video_feature_path):
            with open(video_feature_path, 'rb') as f:
                video_clip = pickle.load(f)
            video_clip = torch.tensor(video_clip)
            try:
                output_1 = get_model_output(model, processor['video'], tokenizer, video_clip, question_1, args)
                sample_set['pred1'] = output_1

                # Run inference on the video for the second question and add the output to the list
                output_2 = get_model_output(model, processor['video'], tokenizer, video_clip, question_2, args)
                sample_set['pred2'] = output_2

                output_list.append(sample_set)
            except Exception as e:
                print(f"Error processing video file '{video_name}': {e}")


    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
