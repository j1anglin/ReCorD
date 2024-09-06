from __future__ import annotations

import abc
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from my_utils import GaussianSmoothing

class ReCorDCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        attention_probs_mod = self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs_mod, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def create_controller(
    prompts: List[str], cross_attention_kwargs: Dict, num_inference_steps: int, max_iter_to_alter:int, 
    tokenizer, device, attn_res
) -> AttentionControl:
    edit_type = cross_attention_kwargs.get("edit_type", None)
    local_blend_words = cross_attention_kwargs.get("local_blend_words", None)
    equalizer_words = cross_attention_kwargs.get("equalizer_words", None)
    equalizer_strengths = cross_attention_kwargs.get("equalizer_strengths", None)
    n_cross_replace = cross_attention_kwargs.get("n_cross_replace", 0.4)
    n_self_replace = cross_attention_kwargs.get("n_self_replace", 0.4)

    if edit_type == "ReCorD":
        return AttentionReCorD(
            prompts, num_inference_steps, max_iter_to_alter, n_cross_replace, n_self_replace, tokenizer=tokenizer, device=device, attn_res=attn_res
        )


    raise ValueError(f"Edit type {edit_type} not recognized. Use one of: replace, refine, reweight.")


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if ( self.cur_step*2 ) < self.max_iter_to_alter*2 and self.cur_step%2==1:
                h = attn.shape[0]
                attn[h // 2 :] = self.forward(attn[h // 2 :], is_cross, place_in_unet)
            else:
                attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, attn_res=None):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.attn_res = attn_res


class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        #if (self.att_mod and not self.suppression_masks ):
        if self.cur_step < (self.max_iter_to_alter)*2 and self.cur_step % 2 ==1 and self.indices_to_alter and self.bbox:
            self.suppression_masks = self.get_suppression_masks(power_factor=1)
        
    def get_average_attention(self):
        average_attention = self.attention_store
        """
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store
        }
        """
        return average_attention
    
    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention
    
    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res=None):
        super(AttentionStore, self).__init__(attn_res)
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.save_global_store = False
        self.display_tensors=[]
        self.display_indices=[]        
        self.prompt_len = 0
        self.indices_to_alter = []
        self.bbox = []
        self.display_tensors = []
        self.display_indices = [] 
        self.suppression_masks = []

        

    def set_operations(self, prompt_len, indices_to_alter, bbox, display_tensors, display_indices):
        self.prompt_len = prompt_len
        self.indices_to_alter = indices_to_alter
        self.bbox = bbox
        self.display_tensors=display_tensors
        self.display_indices=display_indices 

        print(f"Setting Done.")

        
            
    def shift_attn(self, attn):
        res = int(np.sqrt(attn.shape[-2]))
        cross_map = attn.reshape(-1, res, res, attn.shape[-1])
        factor = 32/res
        new_maps = cross_map.clone()

            
        #for i, operation in enumerate(self.operations):
        source_coor = tuple(int(x / factor) for x in self.source_coor)
        target_coor = tuple(int(x / factor) for x in self.target_coor)                    


        # Extract source and target coordinates
        sx, sy, sw, sh = source_coor
        tx, ty, tw, th = target_coor

        # Ensuring that the dimensions of the source and target areas are the same
        if (sw, sh) != (tw, th):
            raise ValueError("Source and target dimensions must match")                    
        
        #if ("Shift" in operation):                
        for shift_index in self.indices_to_alter:
            if self.cur_step <= 40:
                new_maps[:, :, :, shift_index] = cross_map[:, :, :, shift_index]/ 10
                new_maps[:, sy:sy+sh, sx:sx+sw, shift_index] = cross_map[:, :, :, shift_index].min() 
                new_maps[:, ty:ty+th, tx:tx+tw, :] = cross_map[:, ty:ty+th, tx:tx+tw, :] / 5
                new_maps[:, ty:ty+th, tx:tx+tw, shift_index] = cross_map[:, sy:sy+sh, sx:sx+sw, shift_index] * 10
            """
            if self.cur_step <= 3:
                new_maps[:, :, :, shift_index] = cross_map[:, :, :, shift_index]/ 10
                new_maps[:, sy:sy+sh, sx:sx+sw, shift_index] = cross_map[:, :, :, shift_index].min()
                new_maps[:, ty:ty+th, tx:tx+tw, shift_index] = cross_map[:, sy:sy+sh, sx:sx+sw, shift_index] * 10
            elif self.cur_step <= 20:
                new_maps[:, :, :, shift_index] = cross_map[:, :, :, shift_index]/ 10
                new_maps[:, ty:ty+th, tx:tx+tw, shift_index] = cross_map[:, ty:ty+th, tx:tx+tw, shift_index]* 10
            """
        new_maps = new_maps.reshape(attn.shape)
        return new_maps    
        
    def get_suppression_masks(self, power_factor = 1):
            
        from_where=("up", "down", "mid")
        
        out = []
        attention_maps = self.get_average_attention()
        res = 32
        num_pixels = res ** 2
        
        
        suppression_masks = []
        for index in (self.indices_to_alter):            
            cross_maps = []
            masks = []
            for location in from_where:
                for item in attention_maps[f"{location}_cross"]:
                    if item.shape[1] == num_pixels:    
                        cross_map = item.reshape(1, -1, res, res, item.shape[-1])[0]
                        cross_maps.append(cross_map)


            subject_attention_map = torch.cat(cross_maps, dim=0)
            
            h = subject_attention_map.shape[0]
            subject_attention_map = subject_attention_map[h//2:]
            subject_attention_map = subject_attention_map.sum(0) / subject_attention_map.shape[0]
            subject_attention_map = subject_attention_map[:, :, index]


            #smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=0.5, dim=2).cuda()
            #subject_attention_map = F.pad(subject_attention_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            #subject_attention_map = smoothing(subject_attention_map).squeeze(0).squeeze(0)                      


            suppression_mask = 1 - torch.nn.functional.softmax(subject_attention_map, dim=-1)
            suppression_mask = suppression_mask ** power_factor            
            self.display_tensors.append(suppression_mask.detach().cpu().numpy())
            self.display_indices.append(f"token_{index}_mask")

            suppression_masks.append(suppression_mask.reshape(-1, res**2) )

        return suppression_masks
    
    
    def attention_suppression(self, attn):
              
        from_where=("up", "down", "mid")
        
        out = []
        attention_maps = self.get_average_attention()
        res = int(np.sqrt(attn.shape[-2]))
        num_pixels = res ** 2

        for i, index in enumerate(self.indices_to_alter):
            if self.bbox[i]!=[]:
                box = [max(round(b / (512 / res)), 0) for b in self.bbox[i]]
                x1, y1, x2, y2 = box
                
                cross_map = attn.reshape( -1, res, res, attn.shape[-1] )
                suppression_mask = self.suppression_masks[i]
                suppression_mask = suppression_mask.reshape(1, 1, 32, 32)  
                suppression_mask = F.interpolate(suppression_mask, size=(res, res), mode='bilinear', align_corners=False)    
                suppression_mask = suppression_mask.reshape(-1, res**2)  


                h = cross_map.shape[0]
                x1, y1, x2, y2 = box
                for j in range(self.prompt_len):
                    
                    if j != index:
                        attn[:, :, j] *= (suppression_mask)
                    """
                    if j == index:
                        cross_map[h//2:, :, :, j] /= 5  
                        cross_map[h//2:, y1:y2, x1:x2, j] *= 5   
                        attn = cross_map.reshape(attn.shape)  
                    else:
                        attn[:, :, j] *= (suppression_mask)
                    """
                    
        
        return attn
    
    

class LocalBlend:
    def __call__(self, x_t, attention_store):
        # note that this code works on the latent level!
        k = 1
        # maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]  # These are the numbers because we want to take layers that are 256 x 256, I think this can be changed to something smarter...like, get all attentions where thesecond dim is self.attn_res[0] * self.attn_res[1] in up and down cross.
        maps = [m for m in attention_store["down_cross"] + attention_store["mid_cross"] +  attention_store["up_cross"] if m.shape[1] == self.attn_res[0] * self.attn_res[1]]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, self.attn_res[0], self.attn_res[1], self.max_num_words) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1) # since alpha_layers is all 0s except where we edit, the product zeroes out all but what we change. Then, the sum adds the values of the original and what we edit. Then, we average across dim=1, which is the number of layers.
        mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = F.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)

        mask = mask[:1] + mask[1:]
        mask = mask.to(torch.float16)

        x_t = x_t[:1] + mask * (x_t - x_t[:1]) # x_t[:1] is the original image. mask*(x_t - x_t[:1]) zeroes out the original image and removes the difference between the original and each image we are generating (mostly just one). Then, it applies the mask on the image. That is, it's only keeping the cells we want to generate.
        return x_t

    def __init__(
        self, prompts: List[str], words: [List[List[str]]], tokenizer, device, threshold=0.3, attn_res=None
    ):
        self.max_num_words = 77
        self.attn_res = attn_res

        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, self.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if isinstance(words_, str):
                words_ = [words_]
            for word in words_:
                ind = get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device) # a one-hot vector where the 1s are the words we modify (source and target)
        self.threshold = threshold


class AttentionControlEdit(AttentionStore, abc.ABC):
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= self.attn_res[0]**2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        #super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        if (self.cur_step) < (self.max_iter_to_alter)*2 and self.cur_step % 2 ==1:
            real_step = self.cur_step//2
            if is_cross or (self.num_self_replace[0] <= real_step < self.num_self_replace[1]):
                h = attn.shape[0] // (self.batch_size)
                attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
                attn_base, attn_replace = attn[0], attn[1:]
                if is_cross:
                    alpha_words = self.cross_replace_alpha[real_step]
                    attn_replace_new = (
                        self.replace_cross_attention(attn_base, attn_replace) * alpha_words
                        + (1 - alpha_words) * attn_replace
                    )
                    attn[1:] = attn_replace_new
                else:
                    attn[1:] = self.replace_self_attention(attn_base, attn_replace)
                attn = attn.reshape(self.batch_size * h, *attn.shape[2:])

        if ((self.cur_step < (self.max_iter_to_alter)*2 and self.cur_step % 2 ==1) or self.cur_step >= (self.max_iter_to_alter)*2) and self.suppression_masks:
            if is_cross and self.bbox:
                attn = self.attention_suppression(attn)

                
        if attn.shape[1] <= 32**2:  # avoid memory overhead
            self.step_store[key].append(attn)
            
        return attn

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
        self_replace_steps: Union[float, Tuple[float, float]],
        local_blend: Optional[LocalBlend],
        tokenizer,
        device,
        attn_res=None,
    ):
        super(AttentionControlEdit, self).__init__(attn_res=attn_res)
        # add tokenizer and device here

        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, self.tokenizer
        ).to(self.device)
        if isinstance(self_replace_steps, float):
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend



class AttentionReCorD(AttentionControlEdit):
    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(
        self,
        prompts,
        num_steps: int,
        max_iter_to_alter:int, 
        cross_replace_steps: float,
        self_replace_steps: float,
        local_blend: Optional[LocalBlend] = None,
        tokenizer=None,
        device=None,
        attn_res=None
    ):
        super(AttentionReCorD, self).__init__(
            prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device, attn_res
        )
        
        self.max_iter_to_alter = max_iter_to_alter
        self.mapper, alphas = get_refinement_mapper(prompts, self.tokenizer)
        self.mapper, alphas = self.mapper.to(self.device), alphas.to(self.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])



### util functions for all Edits
def update_alpha_time_word(
    alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor] = None
):
    if isinstance(bounds, float):
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
    prompts, num_steps, cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]], tokenizer, max_num_words=77
):
    if not isinstance(cross_replace_steps, dict):
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"], i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


### util functions for LocalBlend and ReplacementEdit
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if isinstance(word_place, str):
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif isinstance(word_place, int):
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


### util functions for ReplacementEdit
def get_replacement_mapper_(x: str, y: str, tokenizer, max_len=77):
    words_x = x.split(" ")
    words_y = y.split(" ")
    if len(words_x) != len(words_y):
        raise ValueError(
            f"attention replacement edit can only be applied on prompts with the same length"
            f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words."
        )
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    # return torch.from_numpy(mapper).float()
    return torch.from_numpy(mapper).to(torch.float16)


def get_replacement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers = []
    for i in range(1, len(prompts)):
        mapper = get_replacement_mapper_(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
    return torch.stack(mappers)


### util functions for ReweightEdit
def get_equalizer(
    text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]], tokenizer
):
    if isinstance(word_select, (int, str)):
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for i, word in enumerate(word_select):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = torch.FloatTensor(values[i])
    return equalizer


### util functions for RefinementEdit
class ScoreParams:
    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch
        else:
            return self.match


def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix


def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i - 1])
            y_seq.append(y[j - 1])
            i = i - 1
            j = j - 1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append("-")
            y_seq.append(y[j - 1])
            j = j - 1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i - 1])
            y_seq.append("-")
            i = i - 1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper(x: str, y: str, tokenizer, max_len=77):
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    score = ScoreParams(0, 1, -1)
    matrix, trace_back = global_align(x_seq, y_seq, score)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    alphas = torch.ones(max_len)
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[: mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0] :] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas


def get_refinement_mapper(prompts, tokenizer, max_len=77):
    x_seq = prompts[0]
    mappers, alphas = [], []
    for i in range(1, len(prompts)):
        mapper, alpha = get_mapper(x_seq, prompts[i], tokenizer, max_len)
        mappers.append(mapper)
        alphas.append(alpha)
    return torch.stack(mappers), torch.stack(alphas)


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int,
                        avg: bool = True) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                #cross_maps = cross_maps[20:]
                #print(f"cross_maps: {cross_maps.shape}")
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    if avg:
        out = out.sum(0) / out.shape[0]
    return out






