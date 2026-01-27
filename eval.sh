# nextqa_mc_test,nextqa_oe_test,mvbench,videomme,videomme_w_subtitle,egoschema_subset

# ==================================== llava video 7b
# HF_ENDPOINT=https://hf-mirror.com \
# CUDA_VISIBLE_DEVICES=7 \
# accelerate launch --num_processes=1 \
# --main_process_port=25005 \
# -m lmms_eval \
# --model llava_vid \
# --model_args pretrained=/data/gys/models/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64 \
# --tasks videomme \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix llava_video_float16 \
# --output_path ./logs_retain_20_percent/llava_video_hieravid_float16_FastDDP/

# ==================================== llava onevision 7b
# ,overwrite=False,force_sample=True,add_time_instruction=True
# HF_ENDPOINT=https://hf-mirror.com \
# CUDA_VISIBLE_DEVICES=7 \
# accelerate launch --num_processes=1 \
# --main_process_port=25005 \
# -m lmms_eval \
# --model llava_onevision \
# --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,max_frames_num=32 \
# --tasks videomme \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix llava_onevision_float16 \
# --output_path ./logs_retain_10_percent/llava_onevision_hieravid_float16_frame32

# ==================================== llava onevision 7b - original model
# HF_ENDPOINT=https://hf-mirror.com \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# accelerate launch --num_processes=8 \
# --main_process_port=25003 \
# -m lmms_eval \
# --model llava_vid \
# --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,max_frames_num=32,overwrite=False,force_sample=True,add_time_instruction=True \
# --tasks egoschema_subset \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix llava_onevision_float16 \
# --output_path ./logs_retain_100_percent/llava_onevision_original_float16_frame32

# ====================================== Qwen2 VL
# Run and exactly reproduce qwen2vl results!
# mme as an example
# pip3 install qwen_vl_utils
# HF_ENDPOINT=https://hf-mirror.com \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# accelerate launch --num_processes=4 --main_process_port=12345 -m lmms_eval \
#     --model qwen2_vl \
#     --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct \
#     --tasks videomme_w_subtitle \
#     --batch_size 1 --log_samples \
#     --log_samples_suffix qwen2-vl \
#     --output_path ./logs_qwen2-vl/origin_30_percent
