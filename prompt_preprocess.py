import csv
import pandas as pd


def split_sentence_vcoco(sentence):
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
    verb_index = is_index + 2
    obj_index = len(words)
    
    return prompts, obj_nouns, subject_index, verb_index, obj_index 


def split_sentence_hico(row):
    sentence = row['Prompt'].replace(".", "")
    words = sentence.split(' ')

    token_indices = row['token indices'].replace("[", '').replace("]", '').split(',')

    subject_index = int (token_indices[0])
    verb_index = int (token_indices[0]) + 2
    obj_index = int (token_indices[1]) + 1
    
    words.insert(subject_index, 'is')
    intransitive_verb_sentence = ' '.join(words[:subject_index+2])
    full_prompt = ' '.join(words)
    obj_nouns = words[obj_index-1]

    prompts = [intransitive_verb_sentence, full_prompt]
    
    return prompts, obj_nouns, subject_index, verb_index, obj_index 


def xlsx_to_csv(xlsx_file_path, csv_file_path):
    fields = ["Intransitive_prompt", "full_prompt", "obj_nouns",  "seeds", "subject_index", "verb_index", "obj_index", "object_bbox", "Updates_object_bbox"]

    data = []
    df = pd.read_excel(xlsx_file_path, usecols=['object', 'Verb', 'Prompt', 'token indices'])
    for index, row in df.iterrows():
        if row['Verb'] != 'no_interaction':

            prompts, obj_nouns, subject_index, verb_index, obj_index  = split_sentence_hico(row)
            
            record = {
                    "Intransitive_prompt": f"{prompts[0]}",
                    "full_prompt": f"{prompts[1]}",
                    "obj_nouns": f"{obj_nouns}",
                    "seeds":"",
                    "subject_index": subject_index,
                    "verb_index": verb_index,
                    "obj_index": obj_index,
                    "object_bbox": "",
                    "Updates_object_bbox": ""
                }


            data.append(record)


    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow(row)




def txt_to_csv(txt_file_path, csv_file_path):
    # 欄位名稱
    fields = ["Intransitive_prompt", "full_prompt", "obj_nouns",  "seeds", "subject_index", "verb_index", "obj_index", "object_bbox", "Updates_object_bbox"]

    # 讀取文本文件並解析資料
    data = []
    with open(txt_file_path, 'r', encoding='utf-8') as txtfile:
        for line in txtfile:
            prompts, obj_nouns, subject_index, verb_index, obj_index   = split_sentence_vcoco(line)
            record = {
                    "Intransitive_prompt": f"{prompts[0]}",
                    "full_prompt": f"{prompts[1]}",
                    "obj_nouns": f"{obj_nouns}",
                    "seeds":"",
                    "subject_index": subject_index,
                    "verb_index": verb_index,
                    "obj_index": obj_index,
                    "object_bbox": "",
                    "Updates_object_bbox": ""
                }
            data.append(record)


    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

"""
txt_file_path = 'prompts_vcoco.txt' 
csv_file_path = 'prompts_vcoco.csv' 
txt_to_csv(txt_file_path, csv_file_path)
df = pd.read_csv('prompts_vcoco.csv')
print(df.head())
"""
 
xlsx_file_path = 'prompts_hicodet.xlsx'
csv_file_path = 'prompts_hicodet.csv'
xlsx_to_csv(xlsx_file_path, csv_file_path)
df = pd.read_csv('prompts_hicodet.csv')
print(df.head())