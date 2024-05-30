from score_cal import VIdeoRoIEval
import json
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='get the test jsonl file')
    parser.add_argument('--file_path', type=str, required=True, help='The test jsonl/json file path')
    return parser
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    file_path = args.file_path
    if file_path.endswith('.jsonl'):
        file = open(file_path, 'r')
        data = [eval(data.strip()) for data in file.readlines()]
        file.close()
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        print('Please use jsonl/json file')
        raise RuntimeError('Invalid file type')
    anno = {}
    for i_data in data:
        video_name = i_data.get('video_name')
        anno[video_name] = {}
        anno[video_name] = i_data
    video_name_list = list(anno.keys())
    for video_name in video_name_list:
        # too long response will make fault
        if len(anno[video_name]['pred']) < 300 and len(anno[video_name]['pred']) > 5:
            pass
        else:
            
            del anno[video_name]   
    print('The input test video number: ', len(video_name_list))
    print('The ultimate test video number: ', len(anno))
    VideoRoIBench = VIdeoRoIEval(anno)
    VideoRoIBench.evaluate()
    score_list = []
    for metric, score in VideoRoIBench.eval.items():
        print(f'{metric}: {score:.3f}')
        score_list.append(round(score, 3))
    print(score_list)