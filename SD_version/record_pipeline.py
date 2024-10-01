from typing import Any, Callable, List
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from processors import *
from my_utils import find_largest_contour_center_coor, create_otsu_mask
from my_utils import process_and_save_images, get_operations_and_modify_indices
from my_utils import GaussianSmoothing
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端
import matplotlib.pyplot as plt
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import torchvision
import torch.distributions as dist
import cv2
import datetime

config_P = 0.2

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# +
from typing import Any, Callable, List
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class ReCorDPipeline(StableDiffusionPipeline):
    r"""
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents. scheduler
        ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    _optional_components = ["safety_checker", "feature_extractor"]
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds

    def _aggregate_and_get_display_attention_map(self, attention_store: AttentionStore,
                                                   indices_to_viz: List[int],
                                                   indices_to_alter: List[int],
                                                   attention_res: int,
                                                   normalize_eot: bool ,
                                                   cur_step: int,
                                                   display_tensors: List[torch.Tensor],
                                                   display_indices: List[torch.Tensor]
                                                   ):
        
        prompt = self.prompt
        if isinstance(self.prompt, list):
            prompt = self.prompt[1]
        tokens = self.tokenizer.encode(prompt)
        decoder = self.tokenizer.decode
        
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0,
            avg = False)
        
        last_idx = -1
        if normalize_eot:
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        
        h = attention_maps.shape[0]
        attention_for_text = attention_maps[h//2:]
        attention_for_text = attention_for_text.sum(0) / attention_for_text.shape[0] 
        attention_for_text = attention_for_text[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text_softmax = torch.nn.functional.softmax(attention_for_text, dim=-1)        
        
        # Shift indices since we removed the first token
        indices_to_viz = [index - 1 for index in indices_to_viz]
        indices_to_alter = [index - 1 for index in indices_to_alter]
        #attributes_belongings = [[num - 1 for num in sublist] for sublist in attributes_belongings]

        for i, index in enumerate(indices_to_viz):
            subject_image = attention_for_text_softmax[:, :, index]
            display_tensors.append(subject_image.detach().cpu().numpy())
            display_indices.append(f"step{cur_step}_{decoder(int(tokens[index+1]))}")

        for i, index in enumerate(indices_to_alter):
            subject_image = attention_for_text_softmax[:, :, index]
            display_tensors.append(subject_image.detach().cpu().numpy())
            display_indices.append(f"step{cur_step}_{decoder(int(tokens[index+1]))}")
            
    def _aggregate_and_compute_attention_mask_per_token(self, 
                                                        attention_store: AttentionStore,
                                                        indices_to_alter: List[int]
                                                   ):

             

        attention_maps_16x16 = aggregate_attention(
            attention_store=attention_store,
            res=16,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
            
        
        last_idx =  - 1
        attention_for_text_16x16 = attention_maps_16x16[:, :, 1:last_idx]
        #attention_for_text_32x32 *= 100
        #attention_for_text_32x32 = torch.nn.functional.softmax(attention_for_text_32x32, dim=-1)
        
        
        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]  
        otsu_masks = []
        source_coors = []
        
        for index in indices_to_alter:

            
            attention_image = attention_for_text_16x16[:, :, index]
            normalized_image = (attention_image - attention_image.min()) / (attention_image.max() - attention_image.min())
            """
            ## Convert the tensor to a PIL image
            transform = transforms.Compose([
                                            transforms.Resize((256, 256)),  
                                            transforms.ToPILImage()
                                        ])
            pil_image = transform(normalized_image.unsqueeze(0))

            # Save the image
            pil_image.save(f'att_map_{index}.png')
            """
            
            
            # Calculate Otsu's threshold
            normalized_image = normalized_image*255
            otsu_mask = create_otsu_mask(normalized_image)  
            
            otsu_mask_numpy = otsu_mask.detach().cpu().numpy()
            center_coor = find_largest_contour_center_coor(otsu_mask_numpy)
            
            otsu_masks.append(otsu_mask_numpy)
            source_coors.append(center_coor)
            
            
        return otsu_masks, source_coors
    
    def _compute_max_attention_per_index(self,
                                         attention_maps: torch.Tensor,
                                         indices_to_alter: List[int],
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_eot: bool = False,
                                         bbox: List[int] = None,
                                         device: torch.device = 'cuda:0',
                                         config=None,
                                         ) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_indices_list_fg = []
        max_indices_list_bg = []
        dist_x = []
        dist_y = []

        cnt = 0
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]

            box = [max(round(b / (512 / image.shape[0])), 0) for b in bbox[cnt]]
            
            x1, y1, x2, y2 = box
            cnt += 1

            # coordinates to masks
            obj_mask = torch.zeros_like(image)
            ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(device)
            obj_mask[y1:y2, x1:x2] = ones_mask
            bg_mask = 1 - obj_mask

            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(device)
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)

            # Inner-Box constraint
            k = (obj_mask.sum() * config.P).long()
            max_indices_list_fg.append((image * obj_mask).reshape(-1).topk(k)[0].mean())

            # Outer-Box constraint
            k = (bg_mask.sum() * config.P).long()
            max_indices_list_bg.append((image * bg_mask).reshape(-1).topk(k)[0].mean())

            # Corner Constraint
            gt_proj_x = torch.max(obj_mask, dim=0)[0]
            gt_proj_y = torch.max(obj_mask, dim=1)[0]
            corner_mask_x = torch.zeros_like(gt_proj_x)
            corner_mask_y = torch.zeros_like(gt_proj_y)

            # create gt according to the number config.L
            N = gt_proj_x.shape[0]
            corner_mask_x[max(box[0] - config.L, 0): min(box[0] + config.L + 1, N)] = 1.
            corner_mask_x[max(box[2] - config.L, 0): min(box[2] + config.L + 1, N)] = 1.
            corner_mask_y[max(box[1] - config.L, 0): min(box[1] + config.L + 1, N)] = 1.
            corner_mask_y[max(box[3] - config.L, 0): min(box[3] + config.L + 1, N)] = 1.
            dist_x.append((F.l1_loss(image.max(dim=0)[0], gt_proj_x, reduction='none') * corner_mask_x).mean())
            dist_y.append((F.l1_loss(image.max(dim=1)[0], gt_proj_y, reduction='none') * corner_mask_y).mean())

        return max_indices_list_fg, max_indices_list_bg, dist_x, dist_y

    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                   indices_to_alter: List[int],
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_eot: bool = False,
                                                   bbox: List[int] = None,
                                                   device: torch.device = 'cuda:0',
                                                   config=None,
                                                   ):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0,
            avg = False)
        
        last_idx = -1
        if normalize_eot:
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        
        h = attention_maps.shape[0]
        attention_maps = attention_maps[h//2:]
        attention_maps = attention_maps.sum(0) / attention_maps.shape[0] 



        max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
            bbox=bbox,
            device=device,
            config=config,
        )
        return max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y

    @staticmethod
    def _compute_loss(max_attention_per_index_fg: List[torch.Tensor], max_attention_per_index_bg: List[torch.Tensor],
                      dist_x: List[torch.Tensor], dist_y: List[torch.Tensor], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses_fg = [max(0, 1. - curr_max) for curr_max in max_attention_per_index_fg]
        losses_bg = [max(0, curr_max) for curr_max in max_attention_per_index_bg]
        loss = sum(losses_fg) + sum(losses_bg) + sum(dist_x) + sum(dist_y)
        if return_losses:
            return max(losses_fg), losses_fg
        else:
            return max(losses_fg), loss
    
    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents    
    
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            indices_to_alter: List[int],
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
            attn_res: int =16,
        
            indices_to_viz: List[int] = [],
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
            bbox: List[int] = None,
            config = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

                The keyword arguments to configure the edit are:
                - edit_type (`str`). The edit type to apply. Can be either of `replace`, `refine`, `reweight`.
                - n_cross_replace (`int`): Number of diffusion steps in which cross attention should be replaced
                - n_self_replace (`int`): Number of diffusion steps in which self attention should be replaced
                - local_blend_words(`List[str]`, *optional*, default to `None`): Determines which area should be
                  changed. If None, then the whole image can be changed.
                - equalizer_words(`List[str]`, *optional*, default to `None`): Required for edit type `reweight`.
                  Determines which words should be enhanced.
                - equalizer_strengths (`List[float]`, *optional*, default to `None`) Required for edit type `reweight`.
                  Determines which how much the words in `equalizer_words` should be enhanced.

            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if config.viz_attention:      
            save_path = config.viz_path / prompt[1]
            save_path.mkdir(exist_ok=True, parents=True)  
            
        object_bbox = None
        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if attn_res is None:
            attn_res = 16
        self.attn_res = attn_res
        
        prompt_len = max(len(self.tokenizer.encode(p)) for p in prompt)
        print(f'cross_attention_kwargs: {cross_attention_kwargs}')
        self.controller = create_controller(
            prompt, cross_attention_kwargs, num_inference_steps, max_iter_to_alter=max_iter_to_alter,
            tokenizer=self.tokenizer, device=self.device, attn_res=self.attn_res
        )
        self.register_attention_control(self.controller)  # add attention controller        
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        latents[1] = latents[0]
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1
        
        
        display_tensors =[]
        display_indices = []          
        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)



        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                if (i == 0 and  bbox):
                    #print(f"source_coors:{source_coors}, target coors: {target_coors}")   


                    self.controller.set_operations ( prompt_len = prompt_len,
                                     indices_to_alter= indices_to_alter,
                                     bbox=bbox,
                                     display_tensors = display_tensors,
                                     display_indices = display_indices               
                                    )                  
                
                
                if ( i == int( num_inference_steps - 1 ) and not bbox and config.pred_box):
                    #print("get_source_mask")   
                    otsu_masks, source_coors = self._aggregate_and_compute_attention_mask_per_token(
                        attention_store=self.controller,
                        indices_to_alter=indices_to_alter
                    )
                    if source_coors[0] != None:
                        x, y, w, h = (i * 32 for i in source_coors[0])
                    else:
                        x, y, w, h = (0, 0, 0, 0)
                    object_bbox = list( (x, y, x+w, y+h) )

                    #print(source_coors[0], " , ", object_bbox)
                    #process_and_save_images(otsu_masks, source_coors, save_path)  
                   
                
                if i < max_iter_to_alter and bbox:

                    with torch.enable_grad():
                        #added_cond_kwargs = {"text_embeds": add_text_embeds[-1].unsqueeze(0), "time_ids": add_time_ids[-1].unsqueeze(0)}
                        latents = latents.clone().detach().requires_grad_(True).to(device)
                        # Forward pass of denoising with text conditioning

                        noise_pred_text = self.unet(latents[-1].unsqueeze(0), t,
                                                    encoder_hidden_states=prompt_embeds[-1].unsqueeze(0)).sample          
                        #print("noise_pred_text: ", noise_pred_text.shape)
                        self.unet.zero_grad()

                        # Get max activation value for each subject token
                        max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y = self._aggregate_and_get_max_attention_per_token(
                            attention_store=self.controller,
                            indices_to_alter=indices_to_alter,
                            attention_res=attention_res,
                            smooth_attentions=smooth_attentions,
                            sigma=sigma,
                            kernel_size=kernel_size,
                            normalize_eot=False,
                            bbox=bbox,
                            device=device,
                            config=config
                        )

                        _, loss = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, dist_x, dist_y)
                        
                        del noise_pred_text
                        torch.cuda.empty_cache()
                        
                        if loss != 0:
                            latents = self._update_latent(latents=latents, loss=loss,
                                                          step_size=scale_factor * np.sqrt(scale_range[i]))
                        print(f'Iteration {i} | Loss: {loss:0.4f}')

                
                #added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}    
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                #print("latent_model_input: ", latent_model_input.shape, "latents: ", latents.shape)
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds ).sample

                if config.viz_attention:
                    self._aggregate_and_get_display_attention_map(
                        attention_store=self.controller,
                        indices_to_viz=indices_to_viz,
                        indices_to_alter=indices_to_alter,
                        attention_res=attention_res,
                        normalize_eot=False,
                        cur_step = i,
                        display_tensors=display_tensors,
                        display_indices=display_indices
                    )                     
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # step callback
                latents = self.controller.step_callback(latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        
        if config.viz_attention:        
            # 9. Plot the attention map
            # Number of images per row
            images_per_row = 6

            # Calculate the number of rows needed
            num_rows = len(display_tensors) // images_per_row + (len(display_tensors) % images_per_row > 0)

            # Plot all the tensors
            fig, axes = plt.subplots(num_rows, images_per_row, figsize=(10, 2 * num_rows))

            # Flatten the axes array for easy iteration
            axes = axes.flatten()

            # Hide the axes for the plots that will not have an image
            for i in range(len(display_tensors), len(axes)):
                axes[i].axis('off')

            # Display the images
            for ax, img, text in zip(axes, display_tensors, display_indices):
                ax.imshow(img)
                ax.set_title(text, fontsize=8)
                ax.axis('off')


            #timestamp = datetime.datetime.now().strftime("%d%H%M%S")

            plt.tight_layout()
            plt.savefig(f'{save_path}/attention_map.png')
            
        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            
        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), object_bbox


    def register_attention_control(self, controller):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                list(reversed(self.unet.config.block_out_channels))[block_id]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(controller=controller, place_in_unet=place_in_unet)

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count


# -


