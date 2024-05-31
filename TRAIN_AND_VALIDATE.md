## Data preparation

### data for training
- Download the stage1 and stage2 image data from [Video LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md), for video data, we provide the generated video clip feature, you can download from [Baidu Disk](https://pan.baidu.com/s/1iRjfDdpXzEtnHybbcjLbDA?pwd=sn5z ).
- Download the stage1 and stage2 training annotations. You can download from [Baidu Disk](https://pan.baidu.com/s/1iRjfDdpXzEtnHybbcjLbDA?pwd=sn5z).
- Download the stage3 data, we provide the generated video clip feature and the required images, you can download from  [Baidu Disk](https://pan.baidu.com/s/1iRjfDdpXzEtnHybbcjLbDA?pwd=sn5z) , for [MeViS](https://codalab.lisn.upsaclay.fr/competitions/15094), [GOT10k](http://got-10k.aitestunion.com/) and [MGIT](http://videocube.aitestunion.com/), we don't provide their images, you can download from their official website.
- Download the stage3 training annotations. You can download from [Baidu Disk](https://pan.baidu.com/s/1iRjfDdpXzEtnHybbcjLbDA?pwd=sn5z).

After downloading all of them, organize the data as follows in ```DATA_ROOT```. 

```Shell
DATA_ROOT
├── llava_image
├── llava_image_tune
├── valley
├── videochatgpt_tune
└── Artemis_data
		├──HC-STVG
		├──MeViS_release
		├──A2D_sentences
		├──VID-sentence
		├──GOT-10k
			├──clip_feature
			└── data/full_data/train				
		├──LaSOT
		└── MGIT
			├──clip_feature
			└── MGIT-Train	
```

### data for validating
- For video, videos and annotations for video qa, you can download from [Video LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md) , we also provide the generated video clip feature, you can download from [Baidu Disk](https://pan.baidu.com/s/1vPZswad5auXlDrmV7JJpdg?pwd=lj8b).
- For Video-ChatGPT benchmark, you can download from [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/quantitative_evaluation/README.md), we also provide the generated video clip feature, you can download .

## Training
Specify your `DATA_ROOT` according to the data preparation.
- Stage 1 pretraining script: [pretrain.sh](scripts/pretrain.sh). 
- Stage 2 tuning script: [finetune.sh](scripts/finetune.sh).
- Stage 3 video referring tuning script: [finetune_ref.sh](scripts/finetune_ref.sh)

## Validating
Our video validation code comes from Video-ChatGPT, thanks for their contribution! 

You can refer to the official repository for validation, but we also provide [off-the-shelf](scripts/eval) scripts.

To load unmerged LoRA weights, you simply need to pass an additional argument `--model-base`, which is the base LLM that is used to train the LoRA weights. 

For VideoRefBench 

### VideoRefBench

​	1. Inference to get the result.

```
bash scripts/eval/run_artemis_bench.sh
```

​	2. Evaluation.

```
python get_score_metric.py --file_path /path/to/test/file.jsonl\json
```

### MSRVTT-QA
1. Inference to get the result.
```Shell
bash scripts/eval/run_qa_msrvtt.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/eval/eval_qa_msrvtt.sh
```

### MSVD-QA
1. Inference to get the result.
```Shell
bash scripts/eval/run_qa_msvd.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/eval/eval_qa_msvd.sh
```

### ActivityNet-QA
1. Inference to get the result.
```Shell
bash scripts/eval/run_qa_activitynet.sh
```

2. GPT-Assistant evaluation.
```Shell
bash scripts/eval/eval_qa_activitynet.sh
```









