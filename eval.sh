# nextqa_mc_test,nextqa_oe_test,mvbench,videomme,videomme_w_subtitle,egoschema_subset

# ==================================== llava video 7b
HF_ENDPOINT=https://hf-mirror.com \
accelerate launch --num_processes=8 \
--main_process_port=25005 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=/data/gys/models/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64 \
--tasks nextqa_mc_test,nextqa_oe_test,mvbench,videomme,videomme_w_subtitle \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision_float16 \
--output_path ./logs_retain_10_percent/llava_video_hieravid_float16