export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2

echo "Cuda visible device: $CUDA_VISIBLE_DEVICES"

video_input="/mnt/netdisk2/gaows/VideoX-Fun-gws-edit/datasets/man.mp4"
video_name="man_ic"
prompt_src="a man is talking in a room"
prompt_tar="a man near a clean and golden wide broad beach near the blue sea, with golden shiny sand and clear blue and pink and orange sky, bright sunny golden sunshine shine on his face."
negative_prompt="blurry, low resolution, poorly drawn face, extra limbs, distorted"

height=544
width=960
k_frames=49
fps=8
partial_edit=true # true | false
# Partial edit means only preserving the masked foreground and generate new background, while full edit means preserving the whole image.
do_preprocess=true # true | false
# do preprocess for the first time only, set to false for repeated runs to save time.
provided_bg=false # true | false
# whether to provide a specific background image for generating, if false, will generate background from text prompt only.
provided_bg_path=""
# if provided_bg is true, please provide the path to the background image here.
transfer_blurring="mid" # low | mid | high
# Forced detail control. high for more detail preserving but less lighting quality. vice versa.

if [ "$partial_edit" = true ]; then
    partial_edit_flag="--partial_edit True"
else
    partial_edit_flag=""
fi

if [ "$transfer_blurring" = "high" ]; then
    transfer_blurring=0.4
elif [ "$transfer_blurring" = "mid" ]; then
    transfer_blurring=0.2
else
    transfer_blurring=0.0
fi

cd "$(cd "$(dirname "$0")" && pwd)"
echo "Changed directory to $(pwd)"


if [ "$do_preprocess" = true ]; then
    echo "--------------------------------------"
    echo "Preprocessing..."
    echo "--------------------------------------"
    python run-preprocess-flow-portal.py \
        --video_input ${video_input} \
        --video_name ${video_name} \
        --k_frames ${k_frames} \
        --height ${height} \
        --width ${width} \
        --fps ${fps} \
        --batch_size 7
    python /mnt/netdisk2/gaows/MatAnyone/inference_matanyone.py \
        -i ./datasets/${video_name}/${video_name}.mp4 \
        -o ./datasets/${video_name}/matanyone \
        -m ./datasets/${video_name}/masks/mask_0000.png \
        -c /mnt/netdisk2/gaows/MatAnyone/pretrained_models/matanyone.pth
    python run-preprocess-flow-portal.py \
        --video_input ${video_input} \
        --video_name ${video_name} \
        --k_frames ${k_frames} \
        --height ${height} \
        --width ${width} \
        --fps ${fps} \
        --batch_size 7 \
        --using_existing_masks ./datasets/${video_name}/matanyone/${video_name}_pha.mp4
    echo "Preprocessing completed."
    echo "--------------------------------------"
fi

if [ "$provided_bg" = true ]; then
    python run-iclight-fbc.py \
        --model_path "/mnt/netdisk2/gaows/VideoX-Fun-pretrained/IC-Light/models/iclight_sd15_fbc.safetensors" \
        --base_path ${PWD} \
        --bg_path ${provided_bg_path} \
        --video_name ${video_name} \
        --prompt "${prompt_tar}" \
        --height ${height} \
        --width ${width} \
        --seed 0 \
        --steps 50 \
        ${partial_edit_flag}
    reference_path="./datasets/${video_name}/${video_name}_reference.png"
    python run-iclight-fc-sdedit.py \
        --model_path "/mnt/netdisk2/gaows/VideoX-Fun-pretrained/IC-Light/models/iclight_sd15_fc.safetensors" \
        --base_path ${PWD} \
        --bg_path ${reference_path} \
        --video_name ${video_name} \
        --prompt "${prompt_tar}" \
        --height ${height} \
        --width ${width} \
        --seed 0 \
        --steps 20 \
        ${partial_edit_flag} \
        --stick_back
else
    python run-iclight-fc.py \
        --model_path "/mnt/netdisk2/gaows/VideoX-Fun-pretrained/IC-Light/models/iclight_sd15_fc.safetensors" \
        --base_path ${PWD} \
        --video_name ${video_name} \
        --prompt "${prompt_tar}" \
        --height ${height} \
        --width ${width} \
        --seed 0 \
        --steps 50 \
        ${partial_edit_flag}
    reference_path="./datasets/${video_name}/${video_name}_reference.png"
    python run-iclight-fbc.py \
        --model_path "/mnt/netdisk2/gaows/VideoX-Fun-pretrained/IC-Light/models/iclight_sd15_fbc.safetensors" \
        --base_path ${PWD} \
        --bg_path ${reference_path} \
        --video_name ${video_name} \
        --prompt "${prompt_tar}" \
        --height ${height} \
        --width ${width} \
        --seed 0 \
        --steps 50 \
        ${partial_edit_flag}
    python run-iclight-fc-sdedit.py \
        --model_path "/mnt/netdisk2/gaows/VideoX-Fun-pretrained/IC-Light/models/iclight_sd15_fc.safetensors" \
        --base_path ${PWD} \
        --bg_path ${reference_path} \
        --video_name ${video_name} \
        --prompt "${prompt_tar}" \
        --height ${height} \
        --width ${width} \
        --seed 0 \
        --steps 20 \
        ${partial_edit_flag} \
        #--stick_back
fi


echo "--------------------------------------"
echo "Running flow portal..."
echo "--------------------------------------"
python run-flow-portal.py \
    --video_name ${video_name} \
    --k_frames ${k_frames} \
    --fps ${fps} \
    --height ${height} \
    --width ${width} \
    --prompt_tar "${prompt_tar}" \
    --prompt_src "${prompt_src}" \
    --negative_prompt "blurry, low resolution, poorly drawn face, extra limbs, distorted" \
    --src_guidance_scale 6.0 \
    --tar_guidance_scale 6.0 \
    --gpu_memory_mode "no_offload" \
    --seed 42 \
    --edit_amplifier 1.0 \
    --cache_times 10 \
    --n_avg 1 \
    ${partial_edit_flag} \
    --accuracy 16 \
    --src_blurring 0.5 \
    --transfer_blurring ${transfer_blurring} \
    --model_name "/mnt/netdisk2/gaows/VideoX-Fun-pretrained/models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"
echo "Flow portal completed."
echo "--------------------------------------"
