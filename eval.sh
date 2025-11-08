#!/bin/bash
# nextqa_mc_test,nextqa_oe_test,mvbench,videomme,videomme_w_subtitle,egoschema_subset

# ==================================== llava video 7b


# 定义需要遍历的lambda数组
lambdas=(0.2 0.6)  # 这里替换为你的实际参数值

# 遍历数组中的每个lambda值
for lambda in "${lambdas[@]}"; do
    echo " ============================== (0.2 0.6 0.8): 开始测试 beta=$lambda"
    
    # 设置环境变量并运行命令，将lambda传入环境变量LAMBDA_PARAM
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    HF_ENDPOINT=https://hf-mirror.com \
    SEGMENT_BETA=$lambda \
    accelerate launch --num_processes=4 \
    --main_process_port=25005 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=/data/gys/models/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64 \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "llava_onevision_float16_beta_${lambda}" \
    --output_path "./logs_retain_30_percent/llava_video_ablation_segment_budget_videomme_beta_${lambda}"
    

    echo "beta=$lambda 测试完成"
    echo "----------------------------------------"
done

echo "所有beta参数测试完成"
