import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import random
import sys
import torch.cuda.amp as amp
from tqdm import tqdm
import cv2
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from transformers import T5Tokenizer

from ..models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel, WanTransformer3DModel)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

def resize_partial_mask(mask, latent):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape
    target_size = list(latent_size[2:])
    resized_mask = F.interpolate(
        mask,
        size=target_size,
        mode='trilinear',
        align_corners=False
    )
    # 先对channel维度做平均到1，然后repeat到latent的channel数
    resized_mask = resized_mask.mean(dim=1, keepdim=True)
    resized_mask = resized_mask.repeat(1, latent_size[1], 1, 1, 1)
    return resized_mask


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class WanFunControlPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->clip_image_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DModel,
        clip_image_encoder: CLIPModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, clip_image_encoder=clip_image_encoder, scheduler=scheduler
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spacial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spacial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spacial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spacial_compression_ratio,
            width // self.vae.spacial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_control_latents(
        self, control, control_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if control is not None:
            control = control.to(device=device, dtype=dtype)
            bs = 1
            new_control = []
            for i in range(0, control.shape[0], bs):
                control_bs = control[i : i + bs]
                control_bs = self.vae.encode(control_bs)[0]
                control_bs = control_bs.mode()
                new_control.append(control_bs)
            control = torch.cat(new_control, dim = 0)

        if control_image is not None:
            control_image = control_image.to(device=device, dtype=dtype)
            bs = 1
            new_control_pixel_values = []
            for i in range(0, control_image.shape[0], bs):
                control_pixel_values_bs = control_image[i : i + bs]
                control_pixel_values_bs = self.vae.encode(control_pixel_values_bs)[0]
                control_pixel_values_bs = control_pixel_values_bs.mode()
                new_control_pixel_values.append(control_pixel_values_bs)
            control_image_latents = torch.cat(new_control_pixel_values, dim = 0)
        else:
            control_image_latents = None

        return control, control_image_latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        control_video: Union[torch.FloatTensor] = None,
        control_camera_video: Union[torch.FloatTensor] = None,
        start_image: Union[torch.FloatTensor] = None,
        ref_image: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clip_image: Image = None,
        max_sequence_length: int = 512,
        comfyui_progressbar: bool = False,
        shift: int = 5,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 2)

        # 5. Prepare latents.
        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        if comfyui_progressbar:
            pbar.update(1)

        # Prepare mask latent variables
        if control_camera_video is not None:
            control_latents = None
            # Rearrange dimensions
            # Concatenate and transpose dimensions
            control_camera_latents = torch.concat(
                [
                    torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                    control_camera_video[:, :, 1:]
                ], dim=2
            ).transpose(1, 2)

            # Reshape, transpose, and view into desired shape
            b, f, c, h, w = control_camera_latents.shape
            control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
            control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        elif control_video is not None:
            video_length = control_video.shape[2]
            control_video = self.image_processor.preprocess(rearrange(control_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            control_video = control_video.to(dtype=torch.float32)
            control_video = rearrange(control_video, "(b f) c h w -> b c f h w", f=video_length)
            control_video_latents = self.prepare_control_latents(
                None,
                control_video,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]
            control_camera_latents = None
        else:
            control_video_latents = torch.zeros_like(latents).to(device, weight_dtype)
            control_camera_latents = None

        if start_image is not None:
            video_length = start_image.shape[2]
            start_image = self.image_processor.preprocess(rearrange(start_image, "b c f h w -> (b f) c h w"), height=height, width=width) 
            start_image = start_image.to(dtype=torch.float32)
            start_image = rearrange(start_image, "(b f) c h w -> b c f h w", f=video_length)
            
            start_image_latents = self.prepare_control_latents(
                None,
                start_image,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]

            start_image_latents_conv_in = torch.zeros_like(latents)
            if latents.size()[2] != 1:
                start_image_latents_conv_in[:, :, :1] = start_image_latents
        else:
            start_image_latents_conv_in = torch.zeros_like(latents)

        # Prepare clip latent variables
        if clip_image is not None:
            clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype) 
            clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
        else:
            clip_image = Image.new("RGB", (512, 512), color=(0, 0, 0))  
            clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype) 
            clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
            clip_context = torch.zeros_like(clip_context)

        if self.transformer.config.get("add_ref_conv", False):
            if ref_image is not None:
                video_length = ref_image.shape[2]
                ref_image = self.image_processor.preprocess(rearrange(ref_image, "b c f h w -> (b f) c h w"), height=height, width=width) 
                ref_image = ref_image.to(dtype=torch.float32)
                ref_image = rearrange(ref_image, "(b f) c h w -> b c f h w", f=video_length)
                
                ref_image_latents = self.prepare_control_latents(
                    None,
                    ref_image,
                    batch_size,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_classifier_free_guidance
                )[1]
                ref_image_latents = ref_image_latents[:, :, 0]
            else:
                ref_image_latents = torch.zeros_like(latents)[:, :, 0]
        else:
            if ref_image is not None:
                raise ValueError("The add_ref_conv is False, but ref_image is not None")
            else:
                ref_image_latents = None

        if comfyui_progressbar:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.latent_channels, (num_frames - 1) // self.vae.temporal_compression_ratio + 1, width // self.vae.spacial_compression_ratio, height // self.vae.spacial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1]) 
        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Prepare mask latent variables
                if control_camera_video is not None:
                    control_latents_input = None
                    control_camera_latents_input = (
                        torch.cat([control_camera_latents] * 2) if do_classifier_free_guidance else control_camera_latents
                    ).to(device, weight_dtype)
                else:
                    control_latents_input = (
                        torch.cat([control_video_latents] * 2) if do_classifier_free_guidance else control_video_latents
                    ).to(device, weight_dtype)
                    control_camera_latents_input = None

                start_image_latents_conv_in_input = (
                    torch.cat([start_image_latents_conv_in] * 2) if do_classifier_free_guidance else start_image_latents_conv_in
                ).to(device, weight_dtype)
                control_latents_input = start_image_latents_conv_in_input if control_latents_input is None else \
                    torch.cat([control_latents_input, start_image_latents_conv_in_input], dim = 1)

                clip_context_input = (
                    torch.cat([clip_context] * 2) if do_classifier_free_guidance else clip_context
                )

                if ref_image_latents is not None:
                    full_ref = (
                        torch.cat([ref_image_latents] * 2) if do_classifier_free_guidance else ref_image_latents
                    ).to(device, weight_dtype)
                else:
                    full_ref = None

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                
                # predict noise model_output
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred = self.transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        y=control_latents_input,
                        y_camera=control_camera_latents_input, 
                        full_ref=full_ref,
                        clip_fea=clip_context_input,
                    )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)

        return WanPipelineOutput(videos=video)
    
    def load_video_frames(self, video_path, size=(832, 480)):
        r"""
        Load video frames from the given path and preprocess them.

        Args:
            video_path (str): Path to the video file.
            size (tuple[`int`], *optional*, defaults to (1280,720)): Target resolution for resizing frames.

        Returns:
            torch.Tensor: Tensor of video frames with shape (frame_num, C, H, W).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to target size
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            # Convert to tensor and normalize to [-1, 1]
            frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 127.5 - 1.0
            # Convert to tensor and normailize to [0, 1]
            # frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255
            frames.append(frame)

        cap.release()
        if not frames:
            raise ValueError(f"No frames found in video: {video_path}")

        # Stack frames into a single tensor
        frames_tensor = torch.stack(frames).permute(1, 0, 2, 3).to(self.device)
        latents = self.vae.encode(frames_tensor)
        return latents # [C, F, H, W]
    
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)    
    def flow_portal(
        self,
        prompt_tar: Optional[Union[str, List[str]]] = None,
        prompt_src: Optional[Union[str, List[str]]] = None,
        negative_prompt_tar: Optional[Union[str, List[str]]] = None,
        negative_prompt_src: Optional[Union[str, List[str]]] = None, # 默认和target一样
        height: int = 480,
        width: int = 720,
        control_video_tar: Union[torch.FloatTensor] = None,
        control_video_src: Union[torch.FloatTensor] = None,
        src_video: Union[torch.FloatTensor] = None,

        ref_image_tar: Union[torch.FloatTensor] = None,
        ref_image_src: Union[torch.FloatTensor] = None, # 原始ref图像直接使用frame0
        ref_image_tar_path: str = None,
        ref_image_src_path: str = None,

        num_frames: int = 49,
        seed: int = 0,
        sampling_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6, # 原来的推理参数
        src_guide_scale = 5.0,
        tar_guide_scale = 10.0,

        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,

        prompt_tar_embeds: Optional[torch.FloatTensor] = None,
        prompt_src_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_tar_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_src_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clip_image_src: Image = None,
        clip_image_tar: Image = None,
        max_sequence_length: int = 512,
        comfyui_progressbar: bool = False,
        shift: int = 5,

        visualize: bool = False,
        visualize_dir: str = None,
        partial_edit: bool = False,
        partial_edit_mask: Optional[torch.FloatTensor] = None,

        edit_amplifier: float = 1.0,

        cache_times: int = 0,
        n_avg: int = 1,
        src_blurring: float = 0.0,
        transfer_blurring: float = 0.0,

        warmup: int = 0,

    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """
        #if negative_prompt_src is None:
        #    negative_prompt_src = negative_prompt_tar
        #if negative_prompt_src_embeds is None:
        #    negative_prompt_src_embeds = negative_prompt_tar_embeds

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        F = num_frames
        W, H = width, height

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        torch.manual_seed(seed)

        device = self._execution_device

        target_shape = (self.vae.latent_channels, (num_frames - 1) // self.vae.temporal_compression_ratio + 1, width // self.vae.spacial_compression_ratio, height // self.vae.spacial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1]) 


        # Check inputs. Raise error if not correct
        self.check_inputs(
            prompt_tar,
            height,
            width,
            negative_prompt_tar,
            callback_on_step_end_tensor_inputs,
            prompt_tar_embeds,
            negative_prompt_tar_embeds,
        )
        self.check_inputs(
            prompt_src,
            height,
            width,
            negative_prompt_src,
            callback_on_step_end_tensor_inputs,
            prompt_src_embeds,
            negative_prompt_src_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False


        # 2. Default call parameters
        if prompt_tar is not None and isinstance(prompt_tar, str):
            batch_size = 1
        elif prompt_tar is not None and isinstance(prompt_tar, list):
            batch_size = len(prompt_tar)
        else:
            batch_size = prompt_tar_embeds.shape[0]

        weight_dtype = self.text_encoder.dtype
        print("weight_dtype:", weight_dtype)

        do_classifier_free_guidance = True # hard coding ensure this

        # Encode prompts
        prompt_tar_embeds, negative_prompt_tar_embeds = self.encode_prompt(
            prompt_tar,
            negative_prompt_tar,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_tar_embeds,
            negative_prompt_embeds=negative_prompt_tar_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        in_prompt_tar_embeds = negative_prompt_tar_embeds + prompt_tar_embeds
        prompt_src_embeds, negative_prompt_src_embeds = self.encode_prompt(
            prompt_tar,
            negative_prompt_tar,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_src_embeds,
            negative_prompt_embeds=negative_prompt_src_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        in_prompt_src_embeds = negative_prompt_src_embeds + prompt_src_embeds

        # Prepare timesteps
        #self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.scheduler.set_timesteps(sampling_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Preprocess cotrolling video:
        assert control_video_tar is not None
        assert control_video_src is not None
        video_length = control_video_tar.shape[2]
        control_video_tar = self.image_processor.preprocess(rearrange(control_video_tar, "b c f h w -> (b f) c h w"), height=height, width=width) 
        control_video_tar = control_video_tar.to(dtype=torch.float32)
        control_video_tar = rearrange(control_video_tar, "(b f) c h w -> b c f h w", f=video_length)
        control_video_tar_latents = self.prepare_control_latents(
            None,
            control_video_tar,
            batch_size,
            height,
            width,
            weight_dtype,
            device,
            generator,
            do_classifier_free_guidance
        )[1]
        control_video_src = self.image_processor.preprocess(rearrange(control_video_src, "b c f h w -> (b f) c h w"), height=height, width=width) 
        control_video_src = control_video_src.to(dtype=torch.float32)
        control_video_src = rearrange(control_video_src, "(b f) c h w -> b c f h w", f=video_length)
        control_video_src_latents = self.prepare_control_latents(
            None,
            control_video_src,
            batch_size,
            height,
            width,
            weight_dtype,
            device,
            generator,
            do_classifier_free_guidance
        )[1]
        control_camera_latents = None

        assert ref_image_tar_path is not None
        assert ref_image_src_path is not None
        clip_image_src = Image.open(ref_image_src_path).convert("RGB")
        clip_image_tar = Image.open(ref_image_tar_path).convert("RGB")
        clip_image_src = TF.to_tensor(clip_image_src).sub_(0.5).div_(0.5).to(device, weight_dtype) 
        clip_context_src = self.clip_image_encoder([clip_image_src[:, None, :, :]])
        clip_image_tar = TF.to_tensor(clip_image_tar).sub_(0.5).div_(0.5).to(device, weight_dtype) 
        clip_context_tar = self.clip_image_encoder([clip_image_tar[:, None, :, :]])
 

        assert src_video is not None
        src_video = self.image_processor.preprocess(rearrange(src_video, "b c f h w -> (b f) c h w"), height=height, width=width) 
        src_video = src_video.to(dtype=torch.float32)
        src_video = rearrange(src_video, "(b f) c h w -> b c f h w", f=video_length)
        x_src = self.prepare_control_latents(
            None,
            src_video,
            batch_size,
            height,
            width,
            weight_dtype,
            device,
            generator,
            do_classifier_free_guidance
        )[1]


        # Prepare latents
        import copy
        # Use the provided generator for reproducible noise if given, else default to torch's global RNG
        global_noise_list = []
        for i_avg in range(n_avg):
            torch.manual_seed(generator.initial_seed() + i_avg if hasattr(generator, "initial_seed") else seed + i_avg)
            global_noise = torch.randn_like(x_src)
            global_noise_list.append(global_noise)
        zt_edit_list = []
        for i_avg in range(n_avg):
            zt_edit = copy.deepcopy(global_noise_list[i_avg].detach())
            zt_edit_list.append(zt_edit)

        # # 5. Prepare latents. # 原来的t2v pipeline中latent是随机的
        # latent_channels = self.vae.config.latent_channels
        # latents = self.prepare_latents(
        #     batch_size * num_videos_per_prompt,
        #     latent_channels,
        #     num_frames,
        #     height,
        #     width,
        #     weight_dtype,
        #     device,
        #     generator,
        #     latents,
        # )
        if comfyui_progressbar:
            pbar.update(1)

        start_image_latents_conv_in = torch.zeros_like(x_src)

        # Preprocessing reference image #
        assert ref_image_tar is not None
        assert ref_image_src is not None
        video_length = ref_image_tar.shape[2]
        ref_image_tar = self.image_processor.preprocess(rearrange(ref_image_tar, "b c f h w -> (b f) c h w"), height=height, width=width) 
        ref_image_tar = ref_image_tar.to(dtype=torch.float32)
        ref_image_tar = rearrange(ref_image_tar, "(b f) c h w -> b c f h w", f=video_length)
        ref_image_latents_tar = self.prepare_control_latents(
            None,
            ref_image_tar,
            batch_size,
            height,
            width,
            weight_dtype,
            device,
            generator,
            do_classifier_free_guidance
        )[1]
        ref_image_latents_tar = ref_image_latents_tar[:, :, 0]

        ref_image_src = self.image_processor.preprocess(rearrange(ref_image_src, "b c f h w -> (b f) c h w"), height=height, width=width) 
        ref_image_src = ref_image_src.to(dtype=torch.float32)
        ref_image_src = rearrange(ref_image_src, "(b f) c h w -> b c f h w", f=video_length)
        ref_image_latents_src = self.prepare_control_latents(
            None,
            ref_image_src,
            batch_size,
            height,
            width,
            weight_dtype,
            device,
            generator,
            do_classifier_free_guidance
        )[1]
        ref_image_latents_src = ref_image_latents_src[:, :, 0]
        # no start image
        start_image_latents_conv_in = torch.zeros_like(x_src)

        if comfyui_progressbar:
            pbar.update(1)

        # # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        self._num_timesteps = len(timesteps)
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(sampling_steps + 2)
        

        self.transformer.num_inference_steps = sampling_steps

        with torch.cuda.amp.autocast(dtype=weight_dtype), torch.no_grad():
            cache_latents_list = []
            for _ in range(n_avg):
                cache_latents = None
                cache_latents_list.append(cache_latents)
            cache_current_times = 0
            cache = False

            for index, t in enumerate(tqdm(timesteps)):
                self.transformer.current_steps = index
                t_next = timesteps[timesteps.tolist().index(t) + 1] if t > timesteps[-1] else 0
                t_i = t / 1000.0
                t_im1 = t_next / 1000.0
                if index < warmup:
                    for i_avg in range(n_avg):
                        zt_edit_list[i_avg] = (1 - t_i) * x_src + t_i * global_noise_list[i_avg]
                    if comfyui_progressbar:
                        pbar.update(1)
                    continue
                if cache_current_times == 0:
                    cache = False
                    cache_current_times = cache_times
                    cache_latents_list = []
                else:
                    cache = True
                    cache_current_times -= 1

                z_pred_delta_list = []
                z_pred_tar_list = []
                x_pred_tar_list = []
                for i_avg in range(n_avg):
                    V_delta = None
                    fwd_noise = global_noise_list[i_avg]

                    zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                    zt_tar = zt_edit_list[i_avg]

                    zt_src_latent_model_input = torch.cat([zt_src] * 2)
                    if hasattr(self.scheduler, "scale_model_input"):
                        zt_src_latent_model_input = self.scheduler.scale_model_input(zt_src_latent_model_input, t)
                    zt_tar_latent_model_input = torch.cat([zt_tar] * 2)
                    if hasattr(self.scheduler, "scale_model_input"):
                        zt_tar_latent_model_input = self.scheduler.scale_model_input(zt_tar_latent_model_input, t)
                    
                    control_latents_input_src = (
                        torch.cat([control_video_src_latents] * 2)
                    ).to(device, weight_dtype)
                    control_latents_input_tar = (
                        torch.cat([control_video_tar_latents] * 2)
                    ).to(device, weight_dtype)
                    control_camera_latents_input = None # 不使用，输入None

                    start_image_latents_conv_in_input = (
                        torch.cat([start_image_latents_conv_in] * 2) # 不使用，输入全0
                    ).to(device, weight_dtype)

                    control_latents_input_src =  torch.cat([control_latents_input_src, start_image_latents_conv_in_input], dim = 1)
                    control_latents_input_tar =  torch.cat([control_latents_input_tar, start_image_latents_conv_in_input], dim = 1)

                    clip_context_input_src = torch.cat([clip_context_src] * 2)
                    clip_context_input_tar = torch.cat([clip_context_tar] * 2)

                    full_ref_src = torch.cat([ref_image_latents_src] * 2).to(device, weight_dtype)
                    full_ref_tar = torch.cat([ref_image_latents_tar] * 2).to(device, weight_dtype)

                    no_ref = False
                    if no_ref:
                        full_ref_src = None
                        full_ref_tar = None

                    timestep = t.expand(zt_src_latent_model_input.shape[0])
                    
                    blurring_method = "fft" # gaussian or fft, gaussian means gaussian blur to exclude the high-frequency, fft means fourier transform to exclude the high-frequency
                    
                    if not cache:
                        with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                            with torch.no_grad():
                                noise_pred_src = self.transformer( #
                                    x=zt_src_latent_model_input,
                                    context=in_prompt_src_embeds,
                                    t=timestep,
                                    seq_len=seq_len,
                                    y=control_latents_input_src,
                                    y_camera=control_camera_latents_input,
                                    full_ref=full_ref_src,
                                    clip_fea=clip_context_input_src,
                                )
                                #print("x1:", zt_src_latent_model_input.flatten().tolist()[:5])
                                #print("y1:", control_latents_input_src.flatten().tolist()[:5])
                                #print("f1:", full_ref_src.flatten().tolist()[:5])
                                #print("r1:", noise_pred_src.flatten().tolist()[:5])
                        noise_pred_uncond_src, noise_pred_text_src = noise_pred_src.chunk(2)
                        noise_pred_src= noise_pred_uncond_src + src_guide_scale * (noise_pred_text_src - noise_pred_uncond_src)
                        # source blurring
                        if blurring_method == "gaussian":
                            if src_blurring > 0.0:
                                kernel_size = int(6 * src_blurring + 1)
                                if kernel_size % 2 == 0:
                                    kernel_size += 1
                                sigma = src_blurring
                                B, C, T, H, W = noise_pred_src.shape
                                out = torch.empty_like(noise_pred_src)
                                for b in range(B):
                                    for c in range(C):
                                        for t in range(T):
                                            out[b, c, t] = TF.gaussian_blur(
                                                noise_pred_src[b, c, t].unsqueeze(0), kernel_size=kernel_size, sigma=sigma
                                            ).squeeze(0)
                                noise_pred_src = out
                        elif blurring_method == "fft":
                            if src_blurring > 0.0:
                                orig_dtype = noise_pred_src.dtype
                                noise_pred_src = noise_pred_src.to(torch.float32)
                                B, C, T, H, W = noise_pred_src.shape
                                out = torch.empty_like(noise_pred_src)

                                # 构建一个频率掩膜 (低通滤波器)
                                yy = torch.linspace(-1, 1, H, device=noise_pred_src.device)
                                xx = torch.linspace(-1, 1, W, device=noise_pred_src.device)
                                X, Y = torch.meshgrid(xx, yy, indexing="xy")
                                # 频率半径
                                R = torch.sqrt(X**2 + Y**2)
                                # 高斯低通滤波器
                                cutoff = 1.0 / (src_blurring * 2.0 + 1e-6)  # cutoff频率，值越大，保留的高频成分越多
                                mask = torch.exp(-(R**2) / (2 * (cutoff**2)))  # shape [H, W]

                                for b in range(B):
                                    for c in range(C):
                                        for t in range(T):
                                            img = noise_pred_src[b, c, t]
                                            # 傅里叶变换
                                            f = torch.fft.fft2(img)
                                            fshift = torch.fft.fftshift(f)
                                            # 乘以低通掩膜
                                            fshift_filtered = fshift * mask
                                            # 逆傅里叶
                                            f_ishift = torch.fft.ifftshift(fshift_filtered)
                                            img_back = torch.fft.ifft2(f_ishift).real
                                            out[b, c, t] = img_back
                                noise_pred_src = out.to(orig_dtype)
                        cache_latents_list.append(noise_pred_src)
                    else:
                        noise_pred_src = cache_latents_list[i_avg]

                    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                        with torch.no_grad():
                            noise_pred_tar = self.transformer(
                                x=zt_tar_latent_model_input,
                                context=in_prompt_tar_embeds,
                                t=timestep,
                                seq_len=seq_len,
                                y=control_latents_input_tar,
                                y_camera=control_camera_latents_input,
                                full_ref=full_ref_tar,
                                clip_fea=clip_context_input_tar,
                            )
                            #print("x2:", zt_tar_latent_model_input.flatten().tolist()[:5])
                            #print("y2:", control_latents_input_tar.flatten().tolist()[:5])
                            #print("f2:", full_ref_tar.flatten().tolist()[:5])
                            #print("r2:", noise_pred_tar.flatten().tolist()[:5])
                    noise_pred_uncond_tar, noise_pred_text_tar = noise_pred_tar.chunk(2)
                    noise_pred_tar = noise_pred_uncond_tar + tar_guide_scale * (noise_pred_text_tar - noise_pred_uncond_tar)
                    do_target_blurring = False
                    original_noise_pred_tar = noise_pred_tar
                    if do_target_blurring:
                        if blurring_method == "gaussian":
                            if src_blurring > 0.0:
                                kernel_size = int(6 * src_blurring + 1)
                                if kernel_size % 2 == 0:
                                    kernel_size += 1
                                sigma = src_blurring
                                B, C, T, H, W = noise_pred_tar.shape
                                out = torch.empty_like(noise_pred_tar)
                                for b in range(B):
                                    for c in range(C):
                                        for t in range(T):
                                            out[b, c, t] = TF.gaussian_blur(
                                                noise_pred_tar[b, c, t].unsqueeze(0), kernel_size=kernel_size, sigma=sigma
                                            ).squeeze(0)
                                noise_pred_tar = out
                        elif blurring_method == "fft":
                            if src_blurring > 0.0:
                                orig_dtype = noise_pred_tar.dtype
                                noise_pred_tar = noise_pred_tar.to(torch.float32)
                                B, C, T, H, W = noise_pred_tar.shape
                                out = torch.empty_like(noise_pred_tar)

                                # 构建一个频率掩膜 (低通滤波器)
                                yy = torch.linspace(-1, 1, H, device=noise_pred_tar.device)
                                xx = torch.linspace(-1, 1, W, device=noise_pred_tar.device)
                                X, Y = torch.meshgrid(xx, yy, indexing="xy")
                                # 频率半径
                                R = torch.sqrt(X**2 + Y**2)
                                # 高斯低通滤波器
                                cutoff = 1.0 / (src_blurring * 2.0 + 1e-6)  # cutoff频率，值越大，保留的高频成分越多
                                mask = torch.exp(-(R**2) / (2 * (cutoff**2)))  # shape [H, W]

                                for b in range(B):
                                    for c in range(C):
                                        for t in range(T):
                                            img = noise_pred_tar[b, c, t]
                                            # 傅里叶变换
                                            f = torch.fft.fft2(img)
                                            fshift = torch.fft.fftshift(f)
                                            # 乘以低通掩膜
                                            fshift_filtered = fshift * mask
                                            # 逆傅里叶
                                            f_ishift = torch.fft.ifftshift(fshift_filtered)
                                            img_back = torch.fft.ifft2(f_ishift).real
                                            out[b, c, t] = img_back
                                noise_pred_tar = out.to(orig_dtype)
                        if not partial_edit:
                            pass
                        else:
                            partial_edit_mask = partial_edit_mask.to(device, weight_dtype)
                            partial_edit_mask = resize_partial_mask(partial_edit_mask, x_src)
                            noise_pred_tar = original_noise_pred_tar * (1 - partial_edit_mask) + noise_pred_tar * partial_edit_mask


                    x_pred_src = zt_src - t_i * noise_pred_src
                    x_pred_tar = zt_tar - t_i * noise_pred_tar
                    z_pred_src = zt_src - (t_i - t_im1) * noise_pred_src
                    z_pred_tar = zt_tar - (t_i - t_im1) * noise_pred_tar
                    z_pred_delta = z_pred_tar - z_pred_src
                    retinex_modeling = True
                    if retinex_modeling:
                        x_pred_src_retinex = torch.clamp(x_pred_src + 10, min=8)
                        x_src_retinex = torch.clamp(x_src + 10, min=8)
                        retinex_multiplier = x_src_retinex / x_pred_src_retinex
                        original_range = (z_pred_delta.max(), z_pred_delta.min())
                        z_pred_delta = z_pred_delta * retinex_multiplier
                        z_pred_delta = torch.clamp(z_pred_delta, min=original_range[1], max=original_range[0])
                    z_pred_delta_list.append(z_pred_delta)
                    z_pred_tar_list.append(z_pred_tar)
                    x_pred_tar_list.append(x_pred_tar)
                z_pred_delta_mean = torch.stack(z_pred_delta_list, dim=0).mean(dim=0)
                
                original_z_pred_delta_mean = z_pred_delta_mean
                if blurring_method == "gaussian":
                    if transfer_blurring > 0.0:
                        kernel_size = int(6 * transfer_blurring + 1)
                        if kernel_size % 2 == 0:
                            kernel_size += 1
                        sigma = transfer_blurring
                        B, C, T, H, W = z_pred_delta_mean.shape
                        out = torch.empty_like(z_pred_delta_mean)
                        for b in range(B):
                            for c in range(C):
                                for t in range(T):
                                    out[b, c, t] = TF.gaussian_blur(
                                        z_pred_delta_mean[b, c, t].unsqueeze(0), kernel_size=kernel_size, sigma=sigma
                                    ).squeeze(0)
                        z_pred_delta_mean = out
                elif blurring_method == "fft":
                    if transfer_blurring > 0.0:
                        orig_dtype = z_pred_delta_mean.dtype
                        z_pred_delta_mean = z_pred_delta_mean.to(torch.float32)
                        B, C, T, H, W = z_pred_delta_mean.shape
                        out = torch.empty_like(z_pred_delta_mean)

                        # 构建一个频率掩膜 (低通滤波器)
                        yy = torch.linspace(-1, 1, H, device=z_pred_delta_mean.device)
                        xx = torch.linspace(-1, 1, W, device=z_pred_delta_mean.device)
                        X, Y = torch.meshgrid(xx, yy, indexing="xy")
                        # 频率半径
                        R = torch.sqrt(X**2 + Y**2)
                        # 高斯低通滤波器
                        cutoff = 1.0 / (transfer_blurring * 2.0 + 1e-6)  # cutoff频率，值越大，保留的高频成分越多
                        mask = torch.exp(-(R**2) / (2 * (cutoff**2)))  # shape [H, W]

                        for b in range(B):
                            for c in range(C):
                                for t in range(T):
                                    img = z_pred_delta_mean[b, c, t]
                                    # 傅里叶变换
                                    f = torch.fft.fft2(img)
                                    fshift = torch.fft.fftshift(f)
                                    # 乘以低通掩膜
                                    fshift_filtered = fshift * mask
                                    # 逆傅里叶
                                    f_ishift = torch.fft.ifftshift(fshift_filtered)
                                    img_back = torch.fft.ifft2(f_ishift).real
                                    out[b, c, t] = img_back
                        z_pred_delta_mean = out.to(orig_dtype)
                z_pred_delta_mean = z_pred_delta_mean + 0.5 * (original_z_pred_delta_mean - z_pred_delta_mean)


                for i_avg in range(n_avg):
                    z_pred_edit = (1 - t_im1) * x_src + t_im1 * global_noise_list[i_avg] + z_pred_delta_mean
                    z_pred_edit = z_pred_edit + (edit_amplifier - 1.0) * (z_pred_edit - z_pred_tar_list[i_avg])
                    if not partial_edit:
                        zt_edit_list[i_avg] = z_pred_edit
                    else:
                        partial_edit_mask = partial_edit_mask.to(device, weight_dtype)
                        partial_edit_mask = resize_partial_mask(partial_edit_mask, x_src)
                        zt_edit_list[i_avg] = z_pred_tar * (1 - partial_edit_mask) + z_pred_edit * partial_edit_mask
                    
                    
                    

                tint = int(t_i * 1000)
                try:
                    if visualize and index % 5 == 4:
                        video_temp = self.decode_latents(x_pred_tar_list[0])
                        video_temp = torch.tensor(video_temp)
                        #video_temp = self.video_processor.postprocess_video(video=video_temp, output_type=output_type)
                        save_dir = visualize_dir + f"/video_edit_{tint:04d}"
                        os.makedirs(save_dir, exist_ok=True)
                        video_temp = video_temp.squeeze(0)
                        for i in range(video_temp.shape[1]):  # 遍历帧
                            frame = video_temp[:,i,:,:]  # 取出第 i 帧，shape: (C, H, W)
                            frame_img = TF.to_pil_image(frame.clamp(0, 1).cpu())
                            frame_img.save(os.path.join(save_dir, f"frame_{i:04d}.png"))
                except Exception as e:
                    print(f"Error during visualization: {e}")
            del self.transformer
            torch.cuda.empty_cache()
            with torch.no_grad():
                video = self.decode_latents(zt_edit_list[0])
            video = torch.from_numpy(video)
            #video = self.video_processor.postprocess_video(video, output_type=output_type)
            return video
