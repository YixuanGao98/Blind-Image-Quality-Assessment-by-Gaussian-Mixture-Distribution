# Blind-Image-Quality-Assessment-by-Gaussian-Mixture-Distribution

## checkpoint
Step1: Download [swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth](https://github.com/microsoft/Swin-Transformer) model. Change the path 'swin_model'.

Step2: Download [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) model. Change the path 'vision_tower_name'.

## Run
python main.py --dataset [dataset name] --dataset_path [dataset path] --swin_model [swin_model path] --vision_tower_name [vision_tower_name path]
