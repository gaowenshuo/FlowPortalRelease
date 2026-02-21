# FlowPortal: Residual-Corrected Flow for Training-Free Video Relighting and Background Replacement  
<img src="logo.png" alt="FlowPortal Logo" width="120"/>

[Wenshuo Gao](https://gaowenshuo.github.io/index/), Junyi Fan, Jiangyue Zeng, [Shuai Yang](https://williamyang1991.github.io/)


[🟥 **ArXiv**](https://arxiv.org/abs/2511.18346) · [🟦 **Project Page**](https://gaowenshuo.github.io/FlowPortalProject/)

## News: Accepted by CVPR 2026 !!! 🎇

---

## Abstract

Video relighting with background replacement is a challenging task critical for applications in film production and creative media. Existing methods struggle to balance temporal consistency, spatial fidelity, and illumination naturalness. To address these issues, we introduce FlowPortal, a novel training-free flow-based video relighting framework. Our core innovation is a Residual-Corrected Flow mechanism that transforms a standard flow-based model into an editing model, guaranteeing perfect reconstruction when input conditions are identical and enabling faithful relighting when they differ, resulting in high structural consistency. This is further enhanced by a Decoupled Condition Design for precise lighting control and a High-Frequency Transfer mechanism for detail preservation. Additionally, a masking strategy isolates foreground relighting from background pure generation process. Experiments demonstrate that FlowPortal achieves superior performance in temporal coherence, structural preservation, and lighting realism, while maintaining high efficiency.

---

## Installation

```bash
conda create -n flowportal python=3.10 -y
conda activate flowportal

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install kornia==0.6.9 controlnet-aux xformers==0.0.20 --no-deps
```

### Additional Dependencies

Prepare the following components before running FlowPortal:

1. **MatAnyone**  
   Repository: https://github.com/pq-yang/MatAnyone  
   Download and place the required weights.

2. **IC-Light Weights**  
   Source: https://github.com/lllyasviel/IC-Light/  
   Required files:  
   - `iclight_sd15_fc.safetensors`  
   - `iclight_sd15_fbc.safetensors`

3. **Wan2.1-Fun-V1.1-1.3B-Control Weights**  
   From VideoX-Fun: https://github.com/aigc-apps/VideoX-Fun  
   Place the full directory named:  
   - `Wan2.1-Fun-V1.1-1.3B-Control`

---

## Usage

Run the corresponding bash script:  
```bash
bash run-flow-portal-xx.sh
```

### Arguments

| Parameter | Description |
|----------|-------------|
| `video_input` | Path to the source input video |
| `video_name` | A unique name for the experiment and output directory |
| `prompt_src` | Prompt describing the source video's content |
| `prompt_tar` | Prompt describing target lighting and background |
| `height`, `width` | Output resolution; result will be center-cropped; support multiples of 16 only |
| `fps` | Output frames per second |
| `k_frames` | Number of frames sampled from the input; supports 49 only |
| `partial_edit` | **true**: relight only foreground & generate new background; **false**: full-frame editing |
| `do_preprocess` | Required for first run only; set to false for repeated runs using the same video |
| `provided_bg`, `provided_bg_path` | Use a fixed background image instead of generating one |
| `transfer_blurring` | Controls detail vs. lighting balance (**high** preserves more detail, **low** yield better lighting realism, **mid** is balanced and recommended) |

### Dependency Path Configuration

Ensure the paths to **MatAnyone**, **IC-Light**, and **VideoX-Fun** are correctly updated in the bash scripts before running the system.

---

## Citation (BibTeX)

If you use FlowPortal in your research, please cite:

```bibtex
@article{gao2025flowportal,
  title={FlowPortal: Residual-Corrected Flow for Training-Free Video Relighting and Background Replacement},
  author={Gao, Wenshuo and Fan, Junyi and Zeng, Jiangyue and Yang, Shuai},
  journal={arXiv preprint arXiv:2511.18346},
  year={2025}
}
```

---
