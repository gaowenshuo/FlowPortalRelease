# FlowPortal: Residual-Corrected Flow for Training-Free Video Relighting and Background Replacement

[🟥 **ArXiv**](https://arxiv.org/abs/2511.18346) · [🟦 **Project Page**](https://gaowenshuo.github.io/FlowPortalProject/)


## Abstract

Video relighting with background replacement is a challenging task critical for applications in film production and creative media. Existing methods struggle to balance temporal consistency, spatial fidelity, and illumination naturalness. To address these issues, we introduce FlowPortal, a novel training-free flow-based video relighting framework. Our core innovation is a Residual-Corrected Flow mechanism that transforms a standard flow-based model into an editing model, guaranteeing perfect reconstruction when input conditions are identical and enabling faithful relighting when they differ, resulting in high structural consistency. This is further enhanced by a Decoupled Condition Design for precise lighting control and a High-Frequency Transfer mechanism for detail preservation. Additionally, a masking strategy isolates foreground relighting from background pure generation process. Experiments demonstrate that FlowPortal achieves superior performance in temporal coherence, structural preservation, and lighting realism, while maintaining high efficiency. 

## Installation

```bash

conda create -n flowportal python=3.10 -y

conda activate flowportal

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install kornia==0.6.9 controlnet-aux xformers==0.0.20 --no-deps

```

Then prepare the following dependencies:

1. MatAnyone: https://github.com/pq-yang/MatAnyone and its weights

2. Weights from IC-Light: https://github.com/lllyasviel/IC-Light/
i.e. iclight_sd15_fc.safetensors and iclight_sd15_fbc.safetensors

3. Weights of Wan2.1-Fun-V1.1-1.3B-Control from VideoX-Fun: https://github.com/aigc-apps/VideoX-Fun i.e. a directory named Wan2.1-Fun-V1.1-1.3B-Control

## Usage

Run the bash scripts: run-flow-portal-xx.sh to use FlowPortal.

Parameters:

1. video_input: a path to the source video

2. video_name: a name for the experiment (including output)

3. prompt_src: a prompt describing original input video

4. prompt_tar: a prompt of target video with new lighting and background

5. height / width: the output will be central cropped to this size

6. fps: the fps of output

7. k_frames: the first k frames will be selected from the input video, support 49 only

8. partial_edit: Partial edit means only preserving the masked foreground and generate new background, while full edit means preserving the whole image.

9. do_preprocess: do preprocess for the first time only, set to false for repeated runs to save time, change prompt is ok as long as the same video

10. provided_bg / provided_bg_path: optionally use a given background rather than generated one

11. transfer_blurring: Forced detail control. high for more detail preserving but less lighting quality. vice versa.

Dependency path setting:

The paths of the prepared dependencies (MatAnyone, IC-Light, VideoXFun) should be replacing the current ones in the bash script!
