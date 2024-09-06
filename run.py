import torch
from record_pipeline import ReCorDPipeline
from config import RunConfig
from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image


def split_sentence(sentence):
    sentence = sentence.replace("\n", "")
    words = sentence.split(' ') 
    try:
        is_index = words.index("is")
    except ValueError:
        return "The word 'is' was not found in the sentence."

    intransitive_verb_sentence = ' '.join(words[:is_index+2])
    full_prompt = sentence
    obj_nouns = ' '.join(words[-2:])
    prompts = [intransitive_verb_sentence, full_prompt]
    
    subject_index = is_index
    obj_index = len(words)
    
    
    return prompts, obj_nouns, subject_index, obj_index 



def run_on_prompt(prompts: List[str],
                  model: ReCorDPipeline,
                  cross_attention_kwargs: Optional[Dict[str, Any]],
                  generator: Optional[Union[torch.Generator, List[torch.Generator]]],
                  indices_to_alter: List[list],
                  indices_to_viz: List[list],
                  bbox: List[int],
                  scale_factor:int, 
                  max_iter_to_alter:int,                
                  config: RunConfig) -> Image.Image:

    
    image = model(prompt=prompts, 
                 cross_attention_kwargs=cross_attention_kwargs, 
                 guidance_scale= config.guidance_scale,
                 generator=generator,
                 num_inference_steps=config.n_inference_steps,
                 #denoising_end=config.denoising_end,
                 indices_to_alter=indices_to_alter,
                  indices_to_viz=indices_to_viz,
                 bbox=bbox,
                 scale_factor=scale_factor,
                 max_iter_to_alter=max_iter_to_alter,                  
                  
                 config=config
                )

    return image
