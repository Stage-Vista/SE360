# Copyright 2024 OmniGen team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
warnings.filterwarnings("ignore", message="Ccross_attention_kwargs")

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.SE360.patchembed import OmniGenPatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from einops import rearrange, repeat

class BaseModel(nn.Module):
    def __init__(
        self,
        omnigen_transformer,
    ):
        super().__init__()
        self.omnigen_transformer = omnigen_transformer

        self.patch_embedding = OmniGenPatchEmbed(
            patch_size=2,
            in_channels=4,
            embed_dim=3072,
            pos_embed_max_size=192,
            output_image_proj=None,#self.omnigen_transformer.patch_embedding.output_image_proj,
            input_image_proj=self.omnigen_transformer.patch_embedding.input_image_proj,
        )
        self.trainable_parameters = [(list(self.patch_embedding.output_image_proj.parameters()), 1.0)]


    def _get_multimodal_embeddings(
        self, input_ids: torch.Tensor, input_img_latents: List[torch.Tensor], input_image_sizes: Dict, batch_size: int, num_channels: int, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        if input_ids is None:
            return None

        input_img_latents = [x.to(dtype) for x in input_img_latents]
        erp_input_img_latents = []
        ref_input_img_latents = []
        image_types = []  # Record type of each image for subsequent reconstruction
        
        for latent in input_img_latents:
            if latent.shape[-1] != latent.shape[-2]:  # Width != Height, this is ERP image
                erp_input_img_latents.append(latent)
                image_types.append('erp')
            else:
                ref_input_img_latents.append(latent)
                image_types.append('ref')

        if len(input_img_latents) == 1:
            input_img_latents = input_img_latents[0]
        condition_tokens = self.omnigen_transformer.embed_tokens(input_ids)
        input_img_inx = 0
        
        # Process ERP images and reference images separately
        erp_input_image_tokens = []
        ref_input_image_tokens = []
        
        if len(erp_input_img_latents) > 0:
            erp_input_image_tokens = self.patch_embedding(erp_input_img_latents, is_input_image=True, embed_type="spherical")
            if not isinstance(erp_input_image_tokens, list):
                erp_input_image_tokens = [erp_input_image_tokens]
                
        if len(ref_input_img_latents) > 0:
            ref_input_image_tokens = self.patch_embedding(ref_input_img_latents, is_input_image=True, embed_type="2d")
            if not isinstance(ref_input_image_tokens, list):
                ref_input_image_tokens = [ref_input_image_tokens]
        
        # Reconstruct image tokens in original order
        input_image_tokens = []
        erp_idx = 0
        ref_idx = 0
        
        for img_type in image_types:
            if img_type == 'erp':
                input_image_tokens.append(erp_input_image_tokens[erp_idx])
                erp_idx += 1
            else:  # 'ref'
                input_image_tokens.append(ref_input_image_tokens[ref_idx])
                ref_idx += 1

        if len(input_img_latents) > 0 or input_img_latents is not None:
            for b_inx in input_image_sizes.keys():
                for start_inx, end_inx in input_image_sizes[b_inx]:
                    # replace the placeholder in text tokens with the image embedding.
                    condition_tokens[b_inx, start_inx:end_inx] = input_image_tokens[input_img_inx].to(
                        condition_tokens.dtype
                    )
                    input_img_inx += 1
        return condition_tokens

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.FloatTensor],
        input_ids: torch.Tensor,
        input_img_latents: List[torch.Tensor],
        input_image_sizes: Dict[int, List[int]],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cameras = None,
        return_dict: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        if cameras is not None:
            cameras = {k: rearrange(v, 'b m ... -> (b m) ...') for k, v in cameras.items()}
        if hidden_states is not None:
            if len(hidden_states.shape) == 4:
                batch_size, num_channels, height, width = hidden_states.shape
            elif len(hidden_states.shape) == 5:
                hidden_states = hidden_states.squeeze(1)
                batch_size, num_channels, height, width = hidden_states.shape
        else:
            raise ValueError(f"hidden_states shape is {hidden_states.shape}")
        
        if len(input_img_latents) > 0:   
            input_img_bsize = len(input_img_latents)
        else:
            input_img_bsize = 0
        p = self.omnigen_transformer.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        # 1. Patch & Timestep & Conditional Embedding
        # Determine the expected dtype from the transformer model parameters
        expected_dtype = next(self.omnigen_transformer.parameters()).dtype
        # Cast hidden_states to the expected dtype before passing it to patch_embedding
        hidden_states = hidden_states.to(dtype=expected_dtype)
        # pers_coords_ = get_perscoords_0to1( # Generate spherical coordinates (longitude and latitude) for each pixel in perspective and panoramic images, used to describe geometric projection relationships
        #     height, width, cameras, hidden_states.device,hidden_states.dtype)
        # pers_coords = repeat(pers_coords_, 'b h w c -> (b bb) h w c', bb=batch_size//cameras['height'].size(0))
        # pers_img_coords = repeat(pers_coords_, 'b h w c -> (b bb) h w c', bb=input_img_bsize//cameras['height'].size(0))
        # Before modification:
        # hidden_states = hidden_states + pers_pe
        # pers_coords = rearrange(pers_coords, 'b h w c -> b c h w')
        # pers_img_coords = rearrange(pers_img_coords, 'b h w c -> b c h w')
        # print('pers_coords', pers_coords.shape)
        # After modification: Use torch.cat for channel dimension concatenation
        # hidden_states = torch.cat([hidden_states, pers_coords], dim=1)  # dim=1 is the channel dimension
        # hidden_states = self.new_conv(hidden_states)


        hidden_states = self.patch_embedding(hidden_states, is_input_image=False, embed_type="spherical")
        # hidden_states shape: [batch_size, num_patches, embedding_dim]
        # For example, if input hidden_states was [12, C, H, W] and patch_size is p,
        # num_patches = (H/p) * (W/p).
        # If the output shape is torch.Size([12, 16, 3072]):
        # - 12 is the batch_size.
        # - 16 is the number of patches (sequence_length).
        # - 3072 is the embedding dimension for each patch.
        num_tokens_for_output_image = hidden_states.size(1)

        timestep_proj = self.omnigen_transformer.time_proj(timestep).type_as(hidden_states)
        time_token = self.omnigen_transformer.time_token(timestep_proj).unsqueeze(1)
        temb = self.omnigen_transformer.t_embedder(timestep_proj)
        
        condition_tokens = self._get_multimodal_embeddings(input_ids, input_img_latents, input_image_sizes, input_img_bsize, num_channels, expected_dtype)
        # else:
        #     condition_tokens = self.omnigen_transformer._get_multimodal_embeddings(input_ids, input_img_latents, input_image_sizes)
        
        img_features_for_cat = hidden_states
        len_cond = condition_tokens.size(1) if condition_tokens is not None else 0
        len_time = time_token.size(1)
        len_img_feat = img_features_for_cat.size(1)
        if condition_tokens is not None:
            hidden_states = torch.cat([condition_tokens, time_token, hidden_states], dim=1)
        else:
            hidden_states = torch.cat([time_token, hidden_states], dim=1)

        seq_length = hidden_states.size(1) #251
        position_ids = position_ids.view(-1, seq_length).long() #166

        # 2. Attention mask preprocessing
        if attention_mask is not None and attention_mask.dim() == 3:
            dtype = hidden_states.dtype
            min_dtype = torch.finfo(dtype).min
            attention_mask = (1 - attention_mask) * min_dtype
            attention_mask = attention_mask.unsqueeze(1).type_as(hidden_states)

        # 3. Rotary position embedding
        image_rotary_emb = self.omnigen_transformer.rope(hidden_states, position_ids)

        # 4. Transformer blocks
        for block in self.omnigen_transformer.layers:
            if torch.is_grad_enabled() and self.omnigen_transformer.gradient_checkpointing:
                hidden_states = self.omnigen_transformer._gradient_checkpointing_func(
                    block, hidden_states, attention_mask, image_rotary_emb
                )
            else:
                hidden_states = block(hidden_states, attention_mask=attention_mask, image_rotary_emb=image_rotary_emb)

        # 5. Output norm & projection
        hidden_states = self.omnigen_transformer.norm(hidden_states)  #torch.Size([2, 118, 3072])
        hidden_states = hidden_states[:, -num_tokens_for_output_image:]#torch.Size([2, 32, 3072])
        hidden_states = self.omnigen_transformer.norm_out(hidden_states, temb=temb)
        hidden_states = self.omnigen_transformer.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, post_patch_height, post_patch_width, p, p, -1)
        output = hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
