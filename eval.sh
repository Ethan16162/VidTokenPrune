# nextqa_mc_test,nextqa_oe_test,mvbench,videomme,videomme_w_subtitle,egoschema_subset

# ==================================== llava video 7b
# HF_ENDPOINT=https://hf-mirror.com \
# accelerate launch --num_processes=8 \
# --main_process_port=25005 \
# -m lmms_eval \
# --model llava_onevision \
# --model_args pretrained=/data/gys/models/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64 \
# --tasks nextqa_mc_test,nextqa_oe_test,mvbench,videomme,videomme_w_subtitle \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix llava_video_float16 \
# --output_path ./logs_retain_30_percent/llava_video_hieravid_float16

# ==================================== llava onevision 7b
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes=8 \
--main_process_port=25003 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,max_frames_num=32 \
--tasks videomme_w_subtitle \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision_float16 \
--output_path ./logs_retain_30_percent/llava_onevision_hieravid_float16_frame32