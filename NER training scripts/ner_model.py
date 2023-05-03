# !/usr/bin/python
# coding=utf-8

"""
ner_model.py
By Clarita
- main file to call when running ner model
- calls on data preparation file first before instantiating model
"""


import re
import pandas as pd
import time
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    BertTokenizer,
    BertConfig,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from transformers import DistilBertTokenizerFast, DistilBertConfig
from transformers import DistilBertForTokenClassification
from transformers import BertTokenizerFast, BertForTokenClassification

import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import f1_score

import json
from datasets import load_metric
from sklearn.metrics import classification_report

import json
import argparse
from prepare_data import DataPreparation
from transformers import AutoConfig, AutoModelForTokenClassification

FILE_NAME = ""


def flatten(t):
    return [item for sublist in t for item in sublist]


def compute_metrics(eval_pred):

    metric = load_metric("seqeval")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = [
        [dataset.id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [dataset.id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    report = classification_report(
        flatten(true_labels),
        flatten(true_predictions),
        # labels=list(id2tag.values()),
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()

    report_file = "{}/classification_report.csv".format(model_name + "_NER")
    report_df.to_csv(report_file)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class PrinterCallback(TrainerCallback):
    """
    A bare :class:`~transformers.TrainerCallback` that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
            global_metric_log.append(logs)


def compile_model(pretrained_model_name):

    print("Compiling model")

    # config = AutoConfig.from_pretrained(model_name)
    # config.num_labels = len(dataset.unique_tags)
    # model = AutoModelForTokenClassification.from_config(config)
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name, num_labels=len(dataset.unique_tags)
    )

    print(model)

    outputDir = "{}/results".format(model_name + "_NER")
    training_args = TrainingArguments(
        output_dir=outputDir,  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        learning_rate=2e-5,  # learninng rate
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=10,
        evaluation_strategy="steps",
        report_to="all",
        load_best_model_at_end=True,
        save_steps=30,
        save_total_limit=5,
    )

    return model, training_args


def run_ner(pretrained_model_name, train=True):
    ## Step 2: Compile model
    model, training_args = compile_model(pretrained_model_name)
    print(model)

    trainer = Trainer(
        model=model,  # the instantiated ðŸ¤— Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=dataset.train_dataset,  # training dataset
        eval_dataset=dataset.val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback, EarlyStoppingCallback],
    )

    if train:
        trainer.train()
    else:
        ## Evaluation
        t = trainer.evaluate()
        global_metric_log.append(t)

    ### save logs
    logs_url = training_args.output_dir + "/metric_logs.txt"

    with open(logs_url, "w") as convert_file:
        convert_file.write(json.dumps(global_metric_log))

    print("! Saving trainer model")
    trainer.save_model()

    print("DONE TRAINING!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        nargs=1,
        help="JSON file to be processed",
        type=argparse.FileType("r"),
    )
    arguments = parser.parse_args()
    if arguments.infile is None:
        print("Json file is needed")
    else:
        # Loading a JSON object returns a dict.
        ner_args = json.load(arguments.infile[0])
        print(ner_args)

        augment = ner_args["train_augmentation"]
        training_type = ner_args["training_type"]
        csv_input_type = ner_args["csv_input_type"]
        tagging_scheme = ner_args["tagging_scheme"]

    global_metric_log = []

    model_names = {
        "distilbert": "distilbert-base-uncased",
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
    }

    if training_type == "indiv":
        file_name = ner_args["models"][0]
        pretrained_model_name = model_names[file_name]

        ## 1. prepare dataset
        dataset = DataPreparation(file_name, augment, tagging_scheme)
        if csv_input_type:
            train_ds = ner_args["input_train_csv"]
            val_ds = ner_args["input_test_csv"]
            full_ds = ner_args["input_full_csv"]

            dataset.set_up_splitted_dataset(train_ds, val_ds, full_ds)
        else:
            dataset.set_up_template_dataset(
                ner_args["input_text_file"], "CovidData.csv"
            )

        ## 2. run model
        run_ner(pretrained_model_name)

    elif training_type == "bulk":
        models = ner_args["models"]

        ## bulk training
        ## all model names

        for model_name in models:
            ## 1. prepare dataset
            dataset = DataPreparation(model_name, augment, tagging_scheme)
            if csv_input_type:
                train_ds = ner_args["input_train_csv"]
                val_ds = ner_args["input_test_csv"]
                full_ds = ner_args["input_full_csv"]

                dataset.set_up_splitted_dataset(train_ds, val_ds, full_ds)
            else:
                dataset.set_up_template_dataset(
                    ner_args["input_text_file"], "CovidData.csv"
                )

            file_name = model_names[model_name]
            print(model_name, file_name)
            ## 2. run model
            flag = run_ner(file_name)
