#!/bin/bash

# Anaconda 설치 경로 설정 (Anaconda가 설치된 실제 경로로 변경)
CONDA_PATH="/home/work/source/miniconda3"

# conda 초기화
source "$CONDA_PATH/etc/profile.d/conda.sh"

# 원하는 conda 환경 활성화 (예: myenv)
conda activate kohya_ss

cd /home/work/source/kohya_ss

# 실행할 명령어 (예: Python 스크립트 실행)
CUDA_VISIBLE_DEVICES=1 nohup accelerate launch --num_cpu_threads_per_process=2 "./train_network.py" --enable_bucket --min_bucket_reso=512 --max_bucket_reso=640 --pretrained_model_name_or_path="/home/work/source/ComfyUIHairmake/models/checkpoints/gamma2_finetune_nv_labeled_from_base_e24-000010.safetensors" --train_data_dir="/home/work/source/kohya_ss/dataset/lora/slick_long_no_bangs/img" --resolution="512,640" --output_dir="/home/work/source/kohya_ss/dataset/lora/slick_long_no_bangs/model" --logging_dir="/home/work/source/kohya_ss/dataset/lora/slick_long_no_bangs/log" --network_alpha="16" --save_model_as=safetensors --network_module=networks.lora --text_encoder_lr=5e-05 --unet_lr=0.0001 --network_dim=32 --output_name="dummy" --lr_scheduler_num_cycles="1" --no_half_vae --learning_rate="0.0001" --lr_scheduler="cosine" --lr_warmup_steps="46" --train_batch_size="1" --max_train_epochs="50" --save_every_n_epochs="1000000" --mixed_precision="fp16" --save_precision="fp16" --cache_latents --optimizer_type="AdamW8bit" --max_data_loader_n_workers="0" --bucket_reso_steps=64 --xformers --bucket_no_upscale --noise_offset=0.0 > log_tr_dummy_e40 &

# (선택사항) conda 환경 비활성화
conda deactivate