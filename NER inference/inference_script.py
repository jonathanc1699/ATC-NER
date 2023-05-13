#!/usr/bin/env python
# coding: utf-8

# ## Model Inference Script
# 
# - Deploy existing ner model and tokenizer
# - Evaluate sentences and get entities from the sentences through the ner pipeline
# 


# importing required libraries
import json
from transformers import pipeline
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel


# Making fastapi app
app = FastAPI()


class Item(BaseModel):
    sentence: str


# version should point to the folder containing your trained NER model, it should have tokenizer, id2tag and results files
version = './NER-models/distilbert_NER'
loaded_tokenizer_dir=  "{}/tokenizer".format(version)
id2tag_file = "{}/id2tag.txt".format(version)
loaded_model_dir = "{}/results".format(version)

# opening id2tag_file
with open(id2tag_file, 'r') as convert_file:
    data = convert_file.read()
        
# reconstructing the data as a dictionary
id2tag = json.loads(data)
id2tag = {int(id): tag for id, tag in id2tag.items()}
tag2id = {tag: int(id) for id, tag in id2tag.items()}


# Loading the pre-trained model and tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained(loaded_tokenizer_dir,  local_files_only=True)
loaded_model = AutoModelForTokenClassification.from_pretrained(loaded_model_dir,  local_files_only=True)


# Test request
@app.get("/")
def read_root():
    return {"Hello": "World"}

# NER endpoint - pass sentence; Follow the below format for sending request
# {
#   "sentence": "European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices"
# }
@app.post('/ner_endpoint_1')
def get_ner(data: Item):
    # print(sentence)
    try:
        data = data.dict()
        print(data['sentence'])

        sentence = data['sentence']
        sentence = sentence.strip()
        

        pipe = pipeline('ner', model=loaded_model, tokenizer=loaded_tokenizer)

        features = pipe(sentence)
        features = np.squeeze(features)
        data = []
        for word in features:
            idx = int(word['entity'].split("_")[1])
            word['entity'] = id2tag[idx]
        print(features)
        for words in features:
            entity_data = [words['word'], int(words['start']), words['entity']]
            data.append(entity_data)
        print(data)
        # entity = [word['entity'] for word in features]
        # word = [word['word'] for word in features]
        # for i in range(len(word)):
        #     print(str(word[i]) + " " + str(entity[i]) + "\n")

        return {
            "Error":False,
            "Message": sentence,
            "Data": data
        } 
    except Exception as e:   
        return {
            "Error":True,
            "Message": e,
            "Data": "Error"
        } 
