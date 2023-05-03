"""
prepare_data.py
By Clarita
- file to prepare the given data into train and validation dataset to be fed into the ner model
- input files accepted
(a) csv files -train and test are already split (train.csv, test.csv , full.csv)
(b) txt files -output of the thrax generator (ex. cleaned_covid.txt)

"""
import pandas as pd
import tokenizers
import torch
from transformers import BertTokenizerFast, DistilBertTokenizerFast
import json
import os
from transformers import AutoTokenizer, RobertaTokenizerFast
import re
from sklearn.model_selection import train_test_split

from urllib.request import Request, urlopen  # Python 3


class DataPreparation:
    """
    A class used to represent the Data Preparation module of the NER model pipeline

    Attributes
    ----------
    tokenizer_directory : str
        a string to name the folder where the tokenizer files will be saved (default is model_name)
    tokenizer_file : str
        a string to name the id2tag file (default is model_name)
    augment : bool
        if true, the train dataset is augmented and will require extra parsing and cleaning steps
    model_name : int
        the number of legs the animal has (default 4)
    tokenizer : TokenizerFast (HuggingFace Class)
        tokenizer called in huggingface
    unique_tags: list
        a set of unique tags in the data
    tag2id: dict
        a dictionary that has the tags as key and the id of the tags as values
    id2tag: dict
        a dictionary that has the tags as key and the id of the tags as values
    val_dataset: Dataset
        a dataset that holds the encodings and labels of the text for model evaluation (20% of dataset)
    train_dataset: Dataset
        a dataset that holds the encodings and labels of the text for model training (80% of dataset)
    """

    def __init__(self, model_name, augment, tagging_scheme):
        self.tokenizer_directory = "{}/tokenizer".format(model_name + "_NER")
        self.tokenizer_file = "{}/id2tag.txt".format(model_name + "_NER")
        self.augment = augment
        self.model_name = model_name
        self.tagging_scheme = tagging_scheme
        # tokenizer_file = "{}/id2tag.txt".format(tokenizer_directory)
        # tokenizer_file = "id2tag24.txt"
        self.unique_tags = []
        print("init DataPreparation")
        # self.set_up_splitted_dataset()

    def set_up_template_dataset(self, input_file, outputfile):
        with open(input_file) as f:
            raw_txt = f.readlines()

        final = []
        sentence_id = 0
        for x in raw_txt:
            x = x.lstrip(" ")
            if x[0] != "<":
                continue

            raw_processed = sum(self.get_tagpairs(x, sentence_id), [])
            final.extend(raw_processed)
            sentence_id += 1

        data = pd.DataFrame(final, columns=["Sentence #", "Word", "Tag"])
        data.to_csv(outputfile, index=False)
        print("Created csv file")
        f.close()

        data = pd.read_csv(outputfile)
        data["Tag"] = data["Tag"].apply(self.remove_outside)
        data = data.fillna(method="ffill")

        getter = SentenceGetter(data)
        sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
        labels = [[s[1] for s in sentence] for sentence in getter.sentences]
        train_texts, val_texts, train_tags, val_tags = train_test_split(
            sentences, labels, test_size=0.2
        )

        self.create_dataset(train_texts, train_tags, val_texts, val_tags, data)

    def set_up_splitted_dataset(self, train_ds, val_ds, full_ds):
        ##FOR GMB DATASET ONLY - augmented sentences
        # separate test and train data

        print("Step 1: Setting up datasets")

        # data = pd.read_csv("data/GMB_full.csv", encoding="unicode_escape")
        # test_data = self.fill_na(pd.read_csv("data/GMB_test.csv"))
        data = pd.read_csv(
            full_ds,
            encoding="unicode_escape",
        )
        print(data['Sentence #'])


        test_data = self.fill_na(
            pd.read_csv(
                val_ds,
                encoding="unicode_escape",
            )
        )
        print(test_data['Sentence #'])

        train_data = self.fill_na(pd.read_csv(train_ds, encoding="unicode_escape",))
        print(train_data['Sentence #'])
        ## FOR GMB AUGMENT DATASET

        # if self.augment:
        # train_data = self.fill_na(
        #     pd.read_csv(
        #         "https://raw.githubusercontent.com/whopriyam/Text-Genration-To-Improve-NER-And-Langauge-Modelling/main/data/Augmented_Train_Data.csv"
        #     )
        # )
        # train_data["Tag"] = train_data["Tag"].str.replace("u-", "", regex=False)
        # train_data["Tag"] = train_data["Tag"].str.replace(
        #     "\\bo\\b", "O", regex=True
        # )
        # else:
        #     # train_data = self.fill_na(pd.read_csv("data/GMB_train.csv"))
        #     train_data = self.fill_na(pd.read_csv(train_ds, encoding="unicode_escape"))

        ##get texts and tags
        train_texts, train_tags, val_texts, val_tags = self.prepare_data(
            train_data, test_data
        )

        ## create dataset and encodings
        self.create_dataset(train_texts, train_tags, val_texts, val_tags, data)

    def remove_outside(self, col):
        if col.split("-")[1] == "o":
            return "O"
        else:
            return col

    def get_tagpairs(self, sentence, id):
        sentence = sentence.strip()
        wordlist = re.split(r"(?<=>)(.+?)(?=<)", sentence)
        raw_taglist = wordlist[::4]

        taglist = [x[1:-1] for x in raw_taglist]
        phrases = map(str.strip, wordlist[1::4])

        tagged = map(lambda x: self.tag_items(x[0], x[1], id), zip(taglist, phrases))

        return tagged

    def tag_items(self, tag, phrase, id):
        HEADER = "B-{}".format(tag)
        TRAILER = "L-{}".format(tag)
        MIDDLE = "I-{}".format(tag)
        SINGLE = "U-{}".format(tag)

        words = [x.lower() for x in phrase.split()]

        if tag == "O":
            return [[x, "O"] for x in words]

        if len(words) == 0:
            return [[x, "O"] for x in words]

        if len(words) == 1:
            return [
                [id, words[0], SINGLE],
            ]

        if len(words) == 2:
            return [[id, words[0], HEADER], [id, words[1], TRAILER]]

        intermediaries = [[id, x, MIDDLE] for x in words[1:-1]]

        result = []
        result.extend(
            [
                [id, words[0], HEADER],
            ]
        )
        result.extend(intermediaries)
        result.extend(
            [
                [id, words[-1], TRAILER],
            ]
        )

        return result

    def create_BILOU_tags(self, sentence):
        new_BILOU_tags = []
        for idx in range(0, len(sentence)):
            currTag = sentence[idx][1]

            if idx == 0:
                new_BILOU_tags.append("B-{}".format(currTag))
                prevTag = currTag
            elif idx == len(sentence) - 1:
                if currTag != new_BILOU_tags[-1]:
                    new_BILOU_tags.append("U-{}".format(currTag))
                else:
                    new_BILOU_tags.append("L-{}".format(currTag))
            elif idx > 0 and idx < len(sentence) - 1:
                nextTag = sentence[idx + 1]
                if currTag != prevTag and currTag == nextTag:
                    new_BILOU_tags.append("B-{}".format(currTag))
                elif currTag == prevTag and currTag == nextTag:
                    new_BILOU_tags.append("I-{}".format(currTag))
                elif currTag == prevTag and currTag != nextTag:
                    new_BILOU_tags.append("L-{}".format(currTag))
                elif currTag != nextTag and currTag != nextTag:
                    new_BILOU_tags.append("U-{}".format(currTag))

                prevTag = sentence[idx - 1]

        new_BILOU_tags = [
            "O" if tag in ["B-o", "U-o", "I-o", "L-o"] else tag
            for tag in new_BILOU_tags
        ]
        return new_BILOU_tags

    def create_IOB_tags(self, sentence):
        prev_tag = ""
        tag_list = []
        for s in sentence:
            tag = s[1]
            if tag == "o":
                tag = "O"
            if tag != prev_tag and tag != "O":
                new_tag = "B-{}".format(tag)
            elif tag == prev_tag and tag != "O":
                new_tag = "I-{}".format(tag)
            else:  ## for o tags
                new_tag = tag

            tag_list.append(new_tag)

            prev_tag = tag

        return tag_list

    def fill_na(self, data):
        data = data.fillna(method="ffill")
        return data

    def prepare_data(self, train_data, test_data):
        train_getter = SentenceGetter(train_data)
        test_getter = SentenceGetter(test_data)

        train_texts = [
            [word[0] for word in sentence] for sentence in train_getter.sentences
        ]

        if (
            self.augment
        ):  ##augmented sentences does not have IOB tags and have to be converted
            if self.tagging_scheme == "IOB":
                train_tags = [
                    self.create_IOB_tags(sentence)
                    for sentence in train_getter.sentences
                ]
            elif self.tagging_scheme == "BILOU":
                ##BILOU
                train_tags = [
                    self.create_BILOU_tags(sentence)
                    for sentence in train_getter.sentences
                ]
        else:
            train_tags = [
                [s[1] for s in sentence] for sentence in train_getter.sentences
            ]

        val_texts = [
            [word[0] for word in sentence] for sentence in test_getter.sentences
        ]
        val_tags = [[s[1] for s in sentence] for sentence in test_getter.sentences]

        return train_texts, train_tags, val_texts, val_tags

    def get_tokenizer_encodings(
        self, tokenizer_directory, tokenizer_file, train_texts, val_texts, id2tag
    ):
        # if self.model_name not in os.listdir():
        #     os.makedirs(tokenizer_directory)

        if self.model_name == "distilbert":
            print(self.model_name)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                "distilbert-base-uncased"
            )

        elif self.model_name == "bert":
            self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        elif self.model_name == "roberta":
            self.tokenizer = RobertaTokenizerFast.from_pretrained(
                "roberta-base", add_prefix_space=True
            )

        print(self.tokenizer)

        train_encodings = self.tokenizer(
            train_texts,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )
        val_encodings = self.tokenizer(
            val_texts,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )

        self.tokenizer.save_pretrained(tokenizer_directory)

        # save id2tag file
        with open(tokenizer_file, "w") as convert_file:
            convert_file.write(json.dumps(id2tag))

        convert_file.close()

        return train_encodings, val_encodings

    def create_dataset(self, train_texts, train_tags, val_texts, val_tags, data):
        # train_texts, train_tags, val_texts, val_tags = self.prepare_data(
        #     train_data, test_data
        # )

        if self.augment:
            ##get unique tags of train tags from augmented dataset (have potentially new tags ex U-x)
            train_tags_cpy = []
            for t in train_tags:
                train_tags_cpy.extend(t)

            train_tags_nunique = set(pd.Series(train_tags_cpy).unique())
            full_tags_nunique = set(data["Tag"].unique())

            if len(train_tags_nunique.difference(full_tags_nunique)) > 0:
                self.unique_tags = list(
                    full_tags_nunique.union(
                        train_tags_nunique.difference(full_tags_nunique)
                    )
                )
            else:
                self.unique_tags = set(tag for tag in data["Tag"].values)
        else:
            self.unique_tags = set(tag for tag in data["Tag"].values)

        print("unique tags ", self.unique_tags)

        self.tag2id = {tag: id for id, tag in enumerate(self.unique_tags)}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}

        ## get tokenizer encodings
        train_encodings, val_encodings = self.get_tokenizer_encodings(
            self.tokenizer_directory,
            self.tokenizer_file,
            train_texts,
            val_texts,
            self.id2tag,
        )


        ## list of STRING tag names of each sentence
        tags = train_tags

        ## list of tag IDS of each sentence
        labels = [[self.tag2id[tag] for tag in doc] for doc in tags]

        ## encoded labels with -100 to accomodate the tokenisations
        train_labels = self.encode_tags(train_tags, train_encodings)
        val_labels = self.encode_tags(val_tags, val_encodings)

        train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
        val_encodings.pop("offset_mapping")
        self.train_dataset = Dataset(train_encodings, train_labels)
        self.val_dataset = Dataset(val_encodings, val_labels)

    ##new version
    def encode_tags(self, tags, encodings):

        labels = [[self.tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        i = 0
        for label in labels:
            word_ids = encodings.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    # print('labels' , label)
                    # print('word id' , word_idx)
                    # print('lable[wordid]', label[word_idx])
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            i += 1
            encoded_labels.append(label_ids)

        return encoded_labels


class SentenceGetter(object):
    """
    A class used to create sentences from the data given (in the form of csv where tokens and tags are in dataframes)

    Attributes
    ----------
    data : dataframe
        data
    sentences : list
        list of sentences of the data in the form of tuple(word ,tag)
    """

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [
            (w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())
        ]
        self.grouped = self.data.groupby("Sentence #")
        self.grouped = self.grouped.apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


class Dataset(torch.utils.data.Dataset):

    """
    A class used to create a dataset of the encodings and labels of the train and test set for NER training

    Attributes
    ----------
    encodings : dict
        keys includes 'input_ids', 'attention_mask', 'offset_mapping' of the train or val texts
    labels : str
        the entity tags of each token
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
