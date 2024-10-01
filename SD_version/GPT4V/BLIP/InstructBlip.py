import torch
from PIL import Image
from transformers import (
    InstructBlipVisionConfig,
    InstructBlipQFormerConfig,
    PretrainedConfig,
    OPTConfig,
    InstructBlipConfig,
    InstructBlipForConditionalGeneration,
    InstructBlipVisionModel,
    InstructBlipProcessor,
)


# load sample image
raw_image = Image.open(
    "./hico_20160224_det/images/train2015/HICO_train2015_00000127.jpg").convert("RGB")
# raw_image.show()

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
# model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl", load_in_4bit=True, torch_dtype=torch.float16)
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl", load_in_4bit=True, torch_dtype=torch.float16)
# model.to(device)

# prepare image and prompt for the model
prompt = "What is the interaction between the man and the snowboard?"
inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device=device, dtype=torch.float16)

# autoregressively generate an answer
outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_new_tokens=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
)
# outputs[outputs == 0] = 2
# generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
# print(generated_text)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
