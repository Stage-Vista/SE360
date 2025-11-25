from .PanoGenerator import PanoGenerator
import torch
import os
from PIL import Image
from lightning.pytorch.utilities import rank_zero_only
from ..modules.utils import tensor_to_image
from .Model import BaseModel
import torch.nn.functional as F
from einops import rearrange
import random
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
import inspect
from torchvision import transforms
# # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps= None,
    device = None,
    timesteps = None,
    sigmas = None,
    **kwargs,
):
    r"""
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

class SE360(PanoGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 1. Create model structure
        self.instantiate_model()
        
        # 2. Load checkpoint weights if ckpt_path is provided
        if self.hparams.ckpt_path is not None:
            print(f"Loading checkpoint from {self.hparams.ckpt_path}")
            state_dict = torch.load(self.hparams.ckpt_path, weights_only=True)['state_dict']
            self.convert_state_dict(state_dict)
            self.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded checkpoint weights")
        else:
            print("No checkpoint path provided, using random initialization")
     
        
    def instantiate_model(self):
        pano_omnigen  = self.load_pano()
        self.mv_base_model = BaseModel(pano_omnigen)
        self.trainable_params.extend(self.mv_base_model.trainable_parameters)
        

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    def sample_x0(self,x1):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
        x1 - data point; [batch, *dim]
        """
        if isinstance(x1, (list, tuple)):
            x0 = [torch.randn_like(img_start) for img_start in x1]
        else:
            x0 = torch.randn_like(x1)

        return x0

    def sample_timestep(self,x1):
        u = torch.normal(mean=0.0, std=1.0, size=(len(x1),))
        t = 1 / (1 + torch.exp(-u))
        t = t.to(x1[0])
        return t

    def training_step(self, batch, batch_idx):
        device = batch['pano'].device
        dtype = batch['pano'].dtype
        b = batch['pano'].shape[0]
        
        height = batch['height'][0]
        width = batch['width'][0]
        use_input_image_size_as_output = True
        
        # prepare input and target image lists, process each sample's function type separately
        input_images_list = []
        target_images_list = []
        ref_images_list = []
        pano_mask_list = []
        
        for i in range(b):
            if batch['function'][i] == 'add':
                input_images_list.append(batch['remove_pano'][i])
                target_images_list.append(batch['pano'][i])
                if 'refs' in batch and len(batch['refs']) > 0:
                    ref_images_list.append(batch['refs'][i])
                else:
                    ref_images_list.append(None)
                pano_mask_list.append(batch['pano_mask'][i])
            elif batch['function'][i] == 'remove':
                input_images_list.append(batch['pano'][i])
                target_images_list.append(batch['remove_pano'][i])
                ref_images_list.append(None)  # remove operation does not need reference images
                pano_mask_list.append(batch['pano_mask'][i])
        
        # convert list to tensor
        input_images = torch.stack(input_images_list, dim=0)
        target_images = torch.stack(target_images_list, dim=0)
        pano_mask = torch.stack(pano_mask_list, dim=0)
        
        # process reference images (only needed when add operation exists and reference images exist)
        has_ref_images = any(ref_img is not None for ref_img in ref_images_list)
        if has_ref_images:
            # create zero tensor placeholder for samples without reference images
            ref_images_filled = []
            for ref_img in ref_images_list:
                if ref_img is not None:
                    ref_images_filled.append(ref_img)
                else:
                    # create zero tensor placeholder with the same shape as the first reference image
                    first_ref = next(ref for ref in ref_images_list if ref is not None)
                    ref_images_filled.append(torch.zeros_like(first_ref))
            ref_images = torch.stack(ref_images_filled, dim=0)
        
        input_images = rearrange(input_images, 'b m c h w -> (b m) c h w')
        if has_ref_images:
            ref_images = rearrange(ref_images, 'b m c h w -> (b m) c h w')
            if ref_images.shape[2] != ref_images.shape[3]:
                print(f"Warning: Reference image shape {ref_images.shape} is not right")
        
        target_images = rearrange(target_images, 'b m c h w -> (b m) c h w')
        pano_mask = rearrange(pano_mask, 'b m 1 h w -> (b m) 1 h w')
                  
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            batch['pano_prompt'],
            input_images,
            height,
            width,
            use_input_image_size_as_output,
        )
        
        
        # convert input_images tensor (shape (B, C, H, W), where B = b*m) to a list of PIL.Image objects
        # tensor_to_image should return a (B, H, W, C) NumPy array (uint8)
        
        # for training, we use random probability to decide whether to use image
        # for simplicity, use the same decision for the whole batch
        random_p = torch.rand(1, device=device).item()  # generate single random number
        random_p_ref = torch.rand(1, device=device).item()  # generate single random number
        
        if random_p < self.hparams.image_use_prob:
            numpy_images_bhwc = tensor_to_image(input_images.clone()) # use .clone() to prevent tensor_to_image from modifying the original tensor
            if random_p_ref < self.hparams.ref_use_prob and has_ref_images:
                batch['pano_prompt'] = batch['pano_prompt_with_ref']
                numpy_ref_images_bhwc = tensor_to_image(ref_images.clone())
                pil_images_list = [[Image.fromarray(numpy_images_bhwc[i]), Image.fromarray(numpy_ref_images_bhwc[i])] for i in range(len(numpy_images_bhwc))]
            else:
                pil_images_list = [[Image.fromarray(img_array)] for img_array in numpy_images_bhwc]
            use_img_cfg = True
        else:
            batch['pano_prompt'] = batch['pano_prompt_without_img']
            use_img_cfg = False
            pil_images_list = None

        # pil_images_list now is a list containing b*m PIL.Image objects
        
        processed_data = self.multimodal_processor(
            batch['pano_prompt'],
            pil_images_list, # use converted PIL image list
            height=height,
            width=width,
            use_img_cfg=use_img_cfg,
            use_input_image_size_as_output=False,
            num_images_per_prompt=1,
            mode="train"
        )
        processed_data["input_ids"] = processed_data["input_ids"].to(device)
        processed_data["attention_mask"] = processed_data["attention_mask"].to(device)
        processed_data["position_ids"] = processed_data["position_ids"].to(device)

        if use_img_cfg:
            # pad input images
            padded_input_pixel_values = []
            for img in processed_data["input_pixel_values"]:
                padded_img = self.pad_pano(img)
                padded_input_pixel_values.append(padded_img)
            
            input_img_latents = self.encode_image(padded_input_pixel_values, device=device)
            
            # unpad encoded latents
            unpadded_input_img_latents = []
            for latent in input_img_latents:
                unpadded_latent = self.unpad_pano(latent,latent=True)
                unpadded_input_img_latents.append(unpadded_latent)
            input_img_latents = unpadded_input_img_latents
            # input_img_latents = torch.stack(input_img_latents, dim=0).squeeze(1)

        else:
            input_img_latents = []

        # pil_target_images_list is a list containing PIL.Image objects.
        # we will process each PIL image into a tensor.
        # assume self.multimodal_processor.process_image(img) returns a tensor of shape (C, H, W), e.g. (3, H, W).
        # processed_target_tensors = [
        #     self.multimodal_processor.process_image(pil_img) for pil_img in pil_target_images_list
        # ]
        
        # # stack processed tensors along new batch dimension (dim=0).
        # # if pil_target_images_list has N images, each processed tensor is (C, H, W),
        # # then the shape of stacked_target_images will be (N, C, H, W). this is (batch, 3, height, width).
        # # here N is b*m from the previous re-arrangement operation.
        # stacked_target_images = torch.stack(processed_target_tensors, dim=0)
        
        # # ensure tensors are on the correct device.
        # # data type (dtype) should ideally match what self.encode_image expects (image encoder usually uses float32).
        # # self.multimodal_processor.process_image probably already handled normalization and data type.
        # stacked_target_images = stacked_target_images.to(device=device)

        padded_target_images = self.pad_pano(target_images)
        
        target_img_latents = self.encode_image(padded_target_images, device=device) # the device parameter of encode_image is probably for internal model components.
        
        # unpad encoded latents
        unpadded_target_img_latents = []
        for latent in target_img_latents:
            unpadded_latent = self.unpad_pano(latent,latent=True)
            unpadded_target_img_latents.append(unpadded_latent)
        target_img_latents = torch.stack(unpadded_target_img_latents, dim=0).squeeze(1)


        x0 = self.sample_x0(target_img_latents)
        t = self.sample_timestep(target_img_latents)
        if isinstance(target_img_latents, (list, tuple)):
            # apply .squeeze(0) to each tensor result in the list to remove extra dimensions
            noise_z = [t[i] * target_img_latents[i] for i in range(b)]
            ut = [(target_img_latents[i] - x0[i]) for i in range(b)]
            noise_z = torch.stack(noise_z, dim=0)
            ut = torch.stack(ut, dim=0)
        else:
            dims = [1] * (len(target_img_latents.size()) - 1)
            t_ = t.view(t.size(0), *dims)
            noise_z = t_ * target_img_latents + (1 - t_) * x0
            ut = target_img_latents - x0
        # if use_input_image_size_as_output:
        #     height, width = processed_data["input_pixel_values"][0].shape[-2:]

        # process pano_mask
        random_p_mask = torch.rand(1, device=device).item()
        if random_p_mask < self.hparams.edit_mask_use_prob:
            # downsample to latent space size
            latent_h = noise_z.shape[-2]
            latent_w = noise_z.shape[-1]
            
            # downsample mask to the same space size as noise_z
            pano_mask_downsampled = F.interpolate(
                pano_mask.float(), 
                size=(latent_h, latent_w), 
                mode='nearest'  # use nearest to keep binary mask, avoid generating intermediate values
            )
            
            # concatenate downsampled mask with noise_z in channel dimension
            # noise_z: (b, 4, h, w), pano_mask_downsampled: (b, 1, h, w)
            # 拼接后: (b, 5, h, w)
            noise_z_with_mask = torch.cat([noise_z, pano_mask_downsampled], dim=1)
            use_mask = True
        else:
            # when mask is not used, concatenate a zero mask in channel dimension
            latent_h = noise_z.shape[-2]
            latent_w = noise_z.shape[-1]
            zero_mask = torch.zeros(b, 1, latent_h, latent_w, device=device, dtype=noise_z.dtype)
            noise_z_with_mask = torch.cat([noise_z, zero_mask], dim=1)
            use_mask = False

        denoise = self.mv_base_model(
                    hidden_states=noise_z_with_mask,
                    timestep=t,
                    input_ids=processed_data["input_ids"],
                    input_img_latents=input_img_latents,
                    input_image_sizes=processed_data["input_image_sizes"],
                    attention_mask=processed_data["attention_mask"],
                    position_ids=processed_data["position_ids"],
                    return_dict=False,
                )[0]

        if use_img_cfg:
            # filter out images with unequal width and height (true input images, not reference images)
            # reference images are usually square (equal width and height), input images are rectangular (unequal width and height)
            input_img_latents_for_loss = []
            for latent in input_img_latents:
                if latent.shape[-1] != latent.shape[-2]:  # width != height, this is input image
                    input_img_latents_for_loss.append(latent)
            input_img_latents_for_loss = torch.stack(input_img_latents_for_loss, dim=0).squeeze(1)
            # ensure the length of filtered input images is the same as target_img_latents
            assert len(input_img_latents_for_loss) == len(target_img_latents), \
                f"the number of filtered input images ({len(input_img_latents_for_loss)}) does not match the number of target images ({len(target_img_latents)})"
            
            patch_weight = []
            for i in range(len(target_img_latents)):
                temp_x = target_img_latents[i]
                w = torch.ones_like(temp_x).detach()
                # Find the input image corresponding to the output image. We store the index in need_edit_imgs
                input_x = input_img_latents_for_loss[i]
                
                if input_x.shape != temp_x.shape:
                    print(f"Warning: the {i}th image shape does not match - input: {input_x.shape}, target: {temp_x.shape}")
                    # if shape does not match, set weight to 0
                    w = w * 0
                    patch_weight.append(w)
                    continue
                
                diff = torch.abs(temp_x - input_x).detach() # no grandient for weight
                diff_mean = torch.mean(diff)
                if diff_mean < 0.001:
                    # the difference between the input and output images is too small, so we suspect there might be an issue with this data. We discard the image by setting its weight to zero.
                    w = w * 0
                elif diff_mean <= 0.8:
                    weight = 1 / (diff_mean + 1e-6)
                    weight = max(min(weight, 64), 5) #crop the weight
                    w[diff>0.3] = weight  #assign the weight to the pixels which are different in input and output
                else:
                    # The difference between the input and output images is significant enough, so there's no need to reinforce the loss.
                    pass
                patch_weight.append(w)
            
            # Original loss calculation:
            # loss = torch.nn.functional.mse_loss(ut, denoise, reduction='none')
            # loss = torch.sum(loss * torch.stack(patch_weight, dim=0), dim=1)
            # loss = torch.mean(loss)
            
            squared_error = torch.nn.functional.mse_loss(ut, denoise, reduction='none') # Computes (denoise - ut)**2 element-wise
            
            # stack_patch_weight will have shape (B*M, C, H, W)
            stacked_patch_weights = torch.stack(patch_weight, dim=0)
            
            # Element-wise multiplication with patch weights
            weighted_squared_error = squared_error * stacked_patch_weights
            
            # Mean over all elements, corresponding to mean_flat in the reference
            loss = torch.mean(weighted_squared_error)
        else:
            loss = torch.nn.functional.mse_loss(ut, denoise)
        # loss = torch.nn.functional.mse_loss(ut, denoise)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/loss_pano', loss)
        return loss



    @torch.no_grad()
    def inference(self, batch):
        device = batch['pano'].device
        dtype = batch['pano'].dtype
        b = batch['pano'].shape[0]
        num_cfg = 2
        height = batch['height'][0]
        width = batch['width'][0]
        use_input_image_size_as_output = True
        
        # prepare input image list, process each sample's function type separately
        input_images_list = []
        ref_images_list = []
        
        for i in range(b):
            if batch['function'][i] == 'add':
                input_images_list.append(batch['remove_pano'][i])
                if 'refs' in batch and len(batch['refs']) > 0:
                    ref_images_list.append(batch['refs'][i])
                else:
                    ref_images_list.append(None)
            elif batch['function'][i] == 'remove':
                input_images_list.append(batch['pano'][i])
                ref_images_list.append(None)  # remove operation does not need reference images

        # convert list to tensor
        input_images = torch.stack(input_images_list, dim=0)
        
        # 处理参考图像
        has_ref_images = any(ref_img is not None for ref_img in ref_images_list)
        if has_ref_images:
            # create zero tensor placeholder for samples without reference images
            ref_images_filled = []
            for ref_img in ref_images_list:
                if ref_img is not None:
                    ref_images_filled.append(ref_img)
                else:
                    # create zero tensor placeholder with the same shape as the first reference image
                    first_ref = next(ref for ref in ref_images_list if ref is not None)
                    ref_images_filled.append(torch.zeros_like(first_ref))
            ref_images = torch.stack(ref_images_filled, dim=0)

        input_images = rearrange(input_images, 'b m c h w -> (b m) c h w')
        
        if has_ref_images:
            ref_images = rearrange(ref_images, 'b m c h w -> (b m) c h w')
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            batch['pano_prompt'],
            input_images,
            height,
            width,
            use_input_image_size_as_output,
        )
        
        # convert input_images tensor to a list of PIL.Image objects, and process reference images
        numpy_images_bhwc = tensor_to_image(input_images.clone()) 
        
        # decide processing method based on whether there are reference images
        if has_ref_images and self.hparams.use_ref_in_inference:
            # when there are reference images
            batch['pano_prompt'] = batch.get('pano_prompt_with_ref', batch['pano_prompt'])
            numpy_ref_images_bhwc = tensor_to_image(ref_images.clone())
            pil_images_list = [[Image.fromarray(numpy_images_bhwc[i]), Image.fromarray(numpy_ref_images_bhwc[i])] for i in range(len(numpy_images_bhwc))]
            use_img_cfg = True
            print(f"use ref image in inference")
        else:
            # only input images, no reference images
            pil_images_list = [[Image.fromarray(img_array)] for img_array in numpy_images_bhwc]
            use_img_cfg = True
        
        processed_data = self.multimodal_processor(
            batch['pano_prompt'],
            pil_images_list, # use converted PIL image list
            height=height,
            width=width,
            use_img_cfg=use_img_cfg,
            use_input_image_size_as_output=False,
            num_images_per_prompt=1,
            mode="val"
        )
        processed_data["input_ids"] = processed_data["input_ids"].to(device)
        processed_data["attention_mask"] = processed_data["attention_mask"].to(device)
        processed_data["position_ids"] = processed_data["position_ids"].to(device)

        # encode input images, keep the same processing method as training_step
        if use_img_cfg:
            # pad input images
            padded_input_pixel_values = []
            for img in processed_data["input_pixel_values"]:
                padded_img = self.pad_pano(img)
                padded_input_pixel_values.append(padded_img)
            
            input_img_latents = self.encode_image(padded_input_pixel_values, device=device)
            
            # unpad encoded latents
            unpadded_input_img_latents = []
            for latent in input_img_latents:
                unpadded_latent = self.unpad_pano(latent,latent=True)
                unpadded_input_img_latents.append(unpadded_latent)
            input_img_latents = unpadded_input_img_latents
            # input_img_latents = torch.stack(input_img_latents, dim=0).squeeze(1)

        else:
            input_img_latents = []
            
        sigmas = np.linspace(1, 0, self.hparams.inference_timesteps + 1)[:self.hparams.inference_timesteps]
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, self.hparams.inference_timesteps, device, None, sigmas=sigmas
        )
        self._num_timesteps = len(timesteps)

        latent_channels = self.transformer_channel
        latents = self.prepare_latents(
            b * 1,
            latent_channels,
            height,
            width,
            torch.float32,
            device,
            generator=None,
            latents=None,
        )
        
        curr_rot = 0
        for i, t in enumerate(timesteps):
            # process pano_mask (always use mask if exists in inference)
            if batch.get('pano_mask') is not None and self.hparams.use_mask_in_inference:
                pano_mask = rearrange(batch['pano_mask'], 'b m 1 h w -> (b m) 1 h w')
                # downsample to latent space size
                latent_h = latents.shape[-2]
                latent_w = latents.shape[-1]
                
                # downsample mask to the same space size as latents
                pano_mask_downsampled = F.interpolate(
                    pano_mask.float(), 
                    size=(latent_h, latent_w), 
                    mode='nearest'  # use nearest to keep mask's binary property
                )
                
                # concatenate downsampled mask with latents in channel dimension
                latents_with_mask = torch.cat([latents, pano_mask_downsampled], dim=1)
            else:
                # when mask is not used, concatenate a zero mask in channel dimension
                latent_h = latents.shape[-2]
                latent_w = latents.shape[-1]
                zero_mask = torch.zeros(b, 1, latent_h, latent_w, device=device, dtype=latents.dtype)
                latents_with_mask = torch.cat([latents, zero_mask], dim=1)
            
            # choose forward propagation method based on whether layout condition is used
            latent_model_input = torch.cat([latents_with_mask] * (num_cfg + 1))
            latent_model_input = latent_model_input.to(dtype)
            timestep = t.expand(latent_model_input.shape[0])
            # update latent representation based on noise prediction by scheduler
            noise_pred = self.mv_base_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                input_ids=processed_data["input_ids"],
                input_img_latents=input_img_latents,
                input_image_sizes=processed_data["input_image_sizes"],
                attention_mask=processed_data["attention_mask"],
                position_ids=processed_data["position_ids"],
                return_dict=False,
            )[0]


            # 2. visualize attention maps
            # if i == 0:
            #     self.mv_base_model.visualize_attention_maps(
            #         save_dir="attention_visualizations", 
            #         max_layers=32  # only show first 8 layers
            #     )

            #     # 3. get attention statistics
            #     stats = self.mv_base_model.get_attention_stats()

            #     print("Attention Maps statistics:")
            #     for stat in stats:
            #         print(f"Layer {stat['layer']}: shape={stat['shape']}, "
            #             f"mean={stat['mean']:.4f}, max={stat['max']:.4f}, min={stat['min']:.4f}")
            if num_cfg == 2:
                cond, uncond, img_cond = torch.split(noise_pred, len(noise_pred) // 3, dim=0)
                noise_pred = uncond + self.hparams.image_guidance_scale * (img_cond - uncond) + self.hparams.guidance_scale * (cond - img_cond)
            else:
                cond, uncond = torch.split(noise_pred, len(noise_pred) // 2, dim=0)
                noise_pred = uncond + self.hparams.guidance_scale * (cond - uncond)

            # compute the previous noisy sample x_t -> x_t-1
            # only use first 4 channels (original latents), ignore mask channel
            noise_pred_latents = noise_pred[:, :4]  # extract first 4 channels
            latents = self.scheduler.step(noise_pred_latents, t, latents, return_dict=False)[0]

        latents = latents.to(self.vae.dtype)
        latents = latents / self.vae.config.scaling_factor
        latents = self.pad_pano(latents,latent=True)
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.unpad_pano(image)

        return image

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pano_pred = self.inference(batch)
        
        # process each sample in the mixed batch separately
        for i in range(len(batch['function'])):
            if batch['function'][i] == 'add':
                self.log_val_image(pano_pred[i:i+1], batch['remove_pano'][i:i+1], 
                                 [batch['pano_prompt'][i]], [batch['object_id'][i]],
                                 batch.get('pano_layout_cond')[i:i+1] if batch.get('pano_layout_cond') is not None else None)
            elif batch['function'][i] == 'remove':
                self.log_val_image(pano_pred[i:i+1], batch['pano'][i:i+1], 
                                 [batch['pano_prompt'][i]], [batch['object_id'][i]],
                                 batch.get('pano_layout_cond')[i:i+1] if batch.get('pano_layout_cond') is not None else None)

    def inference_and_save(self, batch, output_dir, ext='png'):
        if len(batch['ref_img_path']) > 0:
            ref_name = batch['ref_img_path'][0].split('/')[-1].split('.')[0]
            if batch['ref_prompt'][0] is not None:
                instruction = batch['ref_prompt'][0].split('.')[0]
                prompt_path = os.path.join(output_dir, f"{batch['view_id'][0]}_{ref_name}_{instruction}.txt")
            else:
                prompt_path = os.path.join(output_dir, f"{batch['view_id'][0]}_{ref_name}.txt")
        else:
            if batch['instruction'][0] is not None:
                instruction = batch['instruction'][0].split('.')[0]
                prompt_path = os.path.join(output_dir, f"{batch['view_id'][0]}_{instruction}.txt")
            else:
                prompt_path = os.path.join(output_dir, f"{batch['view_id'][0]}.txt")
        if os.path.exists(prompt_path):
            return

        pano_pred = self.inference(batch)

        os.makedirs(output_dir, exist_ok=True)
        if len(batch['ref_img_path']) > 0:
            print(f"save ref image in inference")
            if batch['ref_prompt'][0] is not None:
                instruction = batch['ref_prompt'][0].split('.')[0]
                path = os.path.join(output_dir, f"{batch['view_id'][0]}_{ref_name}_{instruction}.{ext}")
            else:
                path = os.path.join(output_dir, f"{batch['view_id'][0]}_{ref_name}.{ext}")
        else:
            if batch['instruction'][0] is not None:
                instruction = batch['instruction'][0].split('.')[0]
                path = os.path.join(output_dir, f"{batch['view_id'][0]}_{instruction}.{ext}")
            else:
                path = os.path.join(output_dir, f"{batch['view_id'][0]}.{ext}")
        # fix BFloat16 type error
        pano_pred_np = tensor_to_image(pano_pred[0])
        im = Image.fromarray(pano_pred_np.squeeze())
        im.save(path)

        with open(prompt_path, 'w') as f:
            f.write(batch['pano_prompt'][0]+'\n')

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, pano_pred, pano, pano_prompt,view_id,
                      pano_layout_cond=None):
        if view_id:
            view_id = view_id[0]
        log_dict = {
            f'val/pano_pred_{view_id}': self.temp_wandb_image(
                pano_pred[0], pano_prompt[0] if pano_prompt else None),
            f'val/pano_gt_{view_id}': self.temp_wandb_image(
                pano[0,0], pano_prompt[0] if pano_prompt else None),
            # f'val/pano_mask_{processed_label}': self.temp_wandb_image(
            #     pano_mask[0, 0], pano_prompt[0] if pano_prompt else None),
        }
        
        if pano_layout_cond is not None:
            log_dict['val/pano_layout_cond'] = self.temp_wandb_image(
                pano_layout_cond[0, 0], pano_prompt[0] if pano_prompt else None)
        
        
        # use wandb to record all information
        self.logger.experiment.log(log_dict)
