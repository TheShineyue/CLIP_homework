# CLIP_homework
This is just a homework about CLIP

At first, you should download 'vit_small_patch16_224' and 'bert-base-uncased'

## Configure the conda environment
pip install -r requirements.txt

## Train CLIP
pyton clip.py --version 2 --train --save_pth model/clip_model_finetuning_v2.pth

## Train CLIP_Next
python clip.py --version 3 --train --save_pth model/clip_model_finetuning_v3.pth

## Only eval
python clip.py --version 3 --model_path model/clip_model_finetuning_v3.pth
