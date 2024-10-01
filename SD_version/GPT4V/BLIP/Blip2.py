import torch
from PIL import Image
# from lavis.models import load_model_and_preprocess
from transformers import Blip2Processor, Blip2ForConditionalGeneration


# load sample image
raw_image = Image.open(
    "./hico_20160224_det/images/train2015/HICO_train2015_00000127.jpg").convert("RGB")
# raw_image.show()

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# ======================================================================================
# Use salesforce's BLIP-2 model

# loads BLIP-2 pre-trained model
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device)
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device)

# # # prepare the image
# image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# print(model.generate(
#     {"image": image, "prompt": "What is the action of the person in the image?"}))

# # prepare context prompt
# context = [
#     ("What is the action of the person in the image?", ""),
#     ("why?", ""),
# ]

# question = "where is the name merlion coming from?"
# template = "Question: {} Answer: {}."
# prompt = " ".join([template.format(context[i][0], context[i][1])
#                   for i in range(len(context))]) + " Question: " + question + " Answer:"
# print(prompt)
# # generate model's response
# model.generate({"image": image, "prompt": prompt})

# ======================================================================================

# Huggingface's BLIP-2 model

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")

model.to(device)

template = "Question: {} Answer: {}."
in_context_learning = [
    ("What is the action of the person in the image?", "The person is snowboarding"),
    # ("What are the keypoints of human pose in the image?", ""),
]

question = "What are the coordinates of of the snowboard in the image?"
prompt = " ".join([template.format(in_context_learning[i][0], in_context_learning[i][1])
                  for i in range(len(in_context_learning))]) + " Question: " + question + " Answer:"
print(prompt)

inputs = processor(raw_image, prompt, return_tensors="pt").to(
    "cuda", torch.float16)

out = model.generate(**inputs, max_new_tokens=500)
generated_text = processor.batch_decode(
    out, skip_special_tokens=True)[0].strip()
print(generated_text)
