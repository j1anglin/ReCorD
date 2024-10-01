import requests
from .data_utils import read_json, encode_image
# from data_utils import read_json, encode_image
# from response import parse_image_number

def gpt4v_single_img(headers, base64_image, p):
    model = "gpt-4-vision-preview"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": p.get()
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    output = response.json()
    print(output)
    prompt_tokens = output['usage']['prompt_tokens']
    completion_tokens = output['usage']['completion_tokens']
    print(f'{prompt_tokens} prompt tokens counted by the OpenAI API.')
    print(f'{completion_tokens} completion tokens counted by the OpenAI API.')
    money_spent = prompt_tokens * 0.00001 + completion_tokens * 0.00003
    print(f"Money spent: ${money_spent}")
    return output


def gpt4v_multi_img(headers, imgs_path_list, prompt):
    model = "gpt-4-vision-preview"
    base64_images = []
    for img in imgs_path_list:
        base64_image = encode_image(img)
        base64_images.append(base64_image)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given the input images and the prompt, {prompt}, which picture contains the most possible pose for the given action? Please answer by number."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[0]}",
                            "detail": "low"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[1]}",
                            "detail": "low"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[2]}",
                            "detail": "low"
                        }
                    },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{base64_images[3]}",
                    #         "detail": "low"
                    #     }
                    # },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{base64_images[4]}",
                    #         "detail": "low"
                    #     }
                    # },
                ]
            }
        ],
        "max_tokens": 100
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    output = response.json()
    print(output)
    prompt_tokens = output['usage']['prompt_tokens']
    completion_tokens = output['usage']['completion_tokens']
    print(f'{prompt_tokens} prompt tokens counted by the OpenAI API.')
    print(f'{completion_tokens} completion tokens counted by the OpenAI API.')
    money_spent = prompt_tokens * 0.00001 + completion_tokens * 0.00003
    print(f"Money spent: ${money_spent}")
    return output

