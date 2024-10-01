import random
from p2pp import p2pp

def generate_subject():
    subjects = ["man", "woman", "boy", "girl", "old man",
                "old woman", "teenager", "child", "young man", "young woman",
                "adult", "kid", "elderly person", "middle-aged person", "toddler"]
    return random.choice(subjects)

def formatting(input_verb, object):
    random_subject = generate_subject()

    def determine_article(word):
        return "an" if word[0].lower() in "aeiou" else "a"

    subject_article = determine_article(random_subject)
    object_article = determine_article(object)
    
    if "_" in input_verb:
        split_verb=input_verb.split("_")
        verb = split_verb[0]
        preposition = split_verb[1]
        verbing = p2pp(verb)
        GT_prompt=f"{subject_article} {random_subject} is {verbing} {preposition} {object_article} {object}"
    else:
        verb = p2pp(input_verb)
        GT_prompt=f"{subject_article} {random_subject} is {verb} {object_article} {object}"
    return GT_prompt

verb = "sitting_on"
object = "apple tree"

GT_prompt = formatting(verb, object)
print(GT_prompt)