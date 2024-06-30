"""
Adapted from : https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus

Running this script:
"""
import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse
import pandas as pd

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=100, type=int)
parser.add_argument("--train_data_path", type=str, required=True)
parser.add_argument("--dev_data_path", type=str, required=True)
parser.add_argument("--dev_data_num_tokens", type=int)
parser.add_argument("--model_name", default='distilbert-base-uncased')
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--num_negatives", default=5, type=int)
parser.add_argument("--control_contriever", default=False, action="store_true")
parser.add_argument("--checkpoint_save_total_limit", default=3, type=int)
parser.add_argument("--batched", default=False, action="store_true")
parser.add_argument("--eval_steps", default=500, type=int)
args = parser.parse_args()

print(args)

# The  model we want to fine-tune
model_name = args.model_name

train_batch_size = args.train_batch_size           #Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory
num_epochs = args.epochs                 # Number of epochs we want to train

# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = 'output/train_bi-encoder-mnrl-{}-{}'.format(model_name.replace("/", "-"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


class RALMDataset(Dataset):
    def __init__(self, json_file_path, batched=False):
        self.data_df = pd.read_json(json_file_path)
        self.batched = batched
        if not self.batched:
            self.triplet_data = []
            all_pos_neg_pairs = []
            for _, data in self.data_df.iterrows():
                dpr_instance = data['dpr_instance']
                pos_neg_pairs = []
                for pos_ctx_idx in dpr_instance['positive_ctxs']:
                    if pos_ctx_idx < len(data['retrieved_docs']):
                        pos_ctx = data['retrieved_docs'][pos_ctx_idx]
                        num_negative_ctxs = 0
                        for neg_ctx_idx in dpr_instance['negative_ctxs']:
                            neg_ctx = data['retrieved_docs'][neg_ctx_idx]
                            if neg_ctx['em'] < pos_ctx['em']:
                            # if num_negative_ctxs < args.num_negatives:
                                if not args.control_contriever or neg_ctx['contriever'] > pos_ctx['contriever']:
                                    self.triplet_data.append((dpr_instance['query'], pos_ctx['text'], neg_ctx['text']))
                                    pos_neg_pairs.append((pos_ctx_idx, neg_ctx_idx))
                                    num_negative_ctxs += 1
                all_pos_neg_pairs.append(pos_neg_pairs)

            self.data_df['pos_neg_pairs'] = all_pos_neg_pairs
            logging.info("Total training pairs: {}".format(len(self.data_df)))

    def __getitem__(self, item):
        if not self.batched:
            query_text, pos_text, neg_text = self.triplet_data[item]

            return InputExample(texts=[query_text, pos_text, neg_text])
        else:
            return self.data_df.iloc[item].to_dict()

    def __len__(self):
        if not self.batched:
            return len(self.triplet_data)
        else:
            return len(self.data_df)





# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = RALMDataset(args.train_data_path, batched=args.batched)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model, similarity_fct=util.dot_score, scale=1)

dev_dataset = RALMDataset(args.dev_data_path, batched=args.batched)


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=0,# len(train_dataloader),
          evaluation_steps=args.eval_steps,
          checkpoint_save_total_limit=args.checkpoint_save_total_limit,
          optimizer_params = {'lr': args.lr},
          save_best_model=True,
          )

# Save the model
model.save(model_save_path)
