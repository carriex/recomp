# adapted from https://github.com/AI21Labs/in-context-ralm/blob/main/eval_lm.py
import os
import argparse
import json
import pickle

import re
import collections

import numpy as np
import torch
import transformers
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
import nltk
import time

from ralm.file_utils import print_args

from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')

import spacy
NER = spacy.load("en_core_web_sm")
# compression methods

def extract_bow(input_str):
    words = collections.Counter(re.findall(r'\w+', input_str))
    bow = []
    for word in words:
        if word not in bow and word not in en_stopwords:
            bow.append(word)
    return bow

def extract_list_of_ner(input_str):
    proccessed_paragraph = NER(input_str)
    return set([ent.text for ent in proccessed_paragraph.ents])


def evaluate_logprob_with_retrieved_docs(
        model,
        tokenizer,
        device,
        encodings,
        begin_loc,
        end_loc,
        trg_len,
        retrieved_item,
        ranking_strategy,
        num_tokens_to_rank,
        retrieval_max_length,
        num_docs=-1,
        compression_method="none",
        top_k=1,
):
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)


    if top_k == 1:
        retrieved_docs = retrieved_item["retrieved_docs"]
    else:
        # concat top k passages to feed in.
        retrieved_docs = [
            {'text': ' '.join([doc['text'] for doc in retrieved_item["retrieved_docs"][i:(i+top_k)]]),
             'score': [doc['score'] for doc in retrieved_item["retrieved_docs"][i:(i+top_k)]],}
            for i in range(len(retrieved_item["retrieved_docs"]) - top_k + 1)
        ]

    num_docs_in_retrieved = len(retrieved_docs)

    if ranking_strategy == "first":
        assert num_docs in [-1, 1], f"In 'first' ranking strategy, unexpected number of docs to rank: {num_docs}"
        num_docs = 1
        chosen_doc_id = 0
    elif ranking_strategy == "random":
        chosen_doc_id = np.random.randint(num_docs_in_retrieved)
        retrieved_docs = [retrieved_docs[chosen_doc_id]]
        num_docs = 1

    num_docs = min(num_docs, num_docs_in_retrieved) if num_docs > 0 else num_docs_in_retrieved

    input_ids = input_ids.repeat(num_docs, 1)
    target_ids = input_ids.clone()
    labels_for_ranking = input_ids.clone()
    assert input_ids.size() == (num_docs, end_loc-begin_loc)

    doc_lens = []




    for doc_id in range(num_docs):
        retrieved_example = retrieved_docs[doc_id]

        doc_title = retrieved_example["title"] if "title" in retrieved_example else None
        doc_text = retrieved_example["text"]
        if doc_title:
            doc_text = doc_title + "\n" + doc_text
        # perform compression
        if compression_method == "bow":
            doc_text = ' '.join(extract_bow(doc_text))
        elif compression_method == "ner":
            doc_text = ' '.join(extract_list_of_ner(doc_text))
        # check doc_text
        encoded_retrieved_text = tokenizer.encode(doc_text, max_length=retrieval_max_length, truncation=True)
        # print(len(encoded_retrieved_text))

        input_ids[doc_id, :len(encoded_retrieved_text)] = torch.tensor(encoded_retrieved_text, device=device)
        doc_lens.append(len(encoded_retrieved_text))

    loss_fct = CrossEntropyLoss(reduction="none")
    per_doc_ranking_loss = None # only for oracle setting

    with torch.no_grad():
        lm_logits = model(input_ids).logits

        # Rank:
        if ranking_strategy in ["first", "random"]:
            batch_doc_id = 0
        else:
            if ranking_strategy == "oracle":
                labels_for_ranking[:, :-trg_len] = -100
                num_tokens_to_rank = trg_len  # We override this variable as it's not really relevant in oracle setting
            else:
                labels_for_ranking[:, :-trg_len-num_tokens_to_rank] = -100
                labels_for_ranking[:, -trg_len:] = -100
            labels_for_ranking = labels_for_ranking[:, 1:]
            assert torch.sum(labels_for_ranking[0] != -100).cpu().item() == num_tokens_to_rank

            lm_logits_for_ranking = lm_logits[..., :-1, :]
            ranking_loss = loss_fct(lm_logits_for_ranking.reshape(-1, lm_logits_for_ranking.size(-1)), labels_for_ranking.reshape(-1))
            ranking_loss = ranking_loss.view(num_docs, -1)
            per_doc_ranking_loss = torch.sum(ranking_loss, dim=-1)
            chosen_doc_id = torch.argmin(per_doc_ranking_loss).cpu().item()
            batch_doc_id = chosen_doc_id

        # Calculate logprob of the chosen doc:
        lm_logits = lm_logits[batch_doc_id, -trg_len-1:-1, :]
        labels = target_ids[batch_doc_id, -trg_len:]
        loss = loss_fct(lm_logits, labels)
        token_ppls = loss.cpu()
        tokens_to_predict = labels.view(-1).cpu().tolist()
        loss = token_ppls.sum()

    per_doc_ranking_loss = per_doc_ranking_loss.cpu().tolist() if torch.is_tensor(per_doc_ranking_loss) else []

    return loss, chosen_doc_id, token_ppls.tolist(), tokens_to_predict, doc_lens[batch_doc_id], per_doc_ranking_loss


def eval_dataset(
        model,
        tokenizer,
        dataset,
        device,
        max_length,
        output_dir=None,
        stride=4,
        normalization_level="word",
        retrieval_dataset=None,
        retrieval_max_length=256,
        ranking_strategy="first",
        num_docs_to_rank=1,
        num_tokens_to_rank_logprob=16,
        compression_method="none",
        top_k=1
):
    encodings = tokenizer(dataset, add_special_tokens=False, return_tensors="pt")

    print("Max context length:", max_length)
    # Number of tokens in dataset
    dataset_len = encodings.input_ids.size(1)
    print("Dataset length:", dataset_len)

    if normalization_level == "word":
        counter = dataset.count(" ")
    elif normalization_level == "token":
        counter = dataset_len
    else:
        raise ValueError(f"Unknown normalization_level: '{normalization_level}'")

    print("Normalization factor (num tokens/words..):", counter)

    nlls = []
    prev_end_loc = 0

    idx = 0
    all_token_ppls = []
    all_tokens_to_predict = []
    all_num_prepended_token = []
    all_chosen_doc_ids = [None]
    all_per_doc_ranking_loss = []
    num_inputs_no_retrieval = 0
    for begin_loc in tqdm(range(0, dataset_len, stride)):
        end_loc = min(begin_loc + max_length, dataset_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        if idx > 0 and retrieval_dataset is not None and len(retrieval_dataset[idx]["retrieved_docs"]) > 0:
            retrieved_example = retrieval_dataset[idx]
            assert retrieved_example["begin_location"] == prev_end_loc
            assert retrieved_example["end_location"] == end_loc

            neg_log_likelihood, chosen_doc_id, token_ppls, tokens_to_predict, num_prepended_token, per_doc_ranking_loss = evaluate_logprob_with_retrieved_docs(
                model, tokenizer, device, encodings, begin_loc, end_loc, trg_len, retrieved_example,
                ranking_strategy=ranking_strategy,
                num_tokens_to_rank=num_tokens_to_rank_logprob,
                retrieval_max_length=retrieval_max_length,
                num_docs=num_docs_to_rank,
                compression_method=compression_method,
                top_k=top_k
            )
            all_chosen_doc_ids.append(chosen_doc_id)
        else:
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            num_prepended_token = 0
            per_doc_ranking_loss = []

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # Calculate per-token loss
                if trg_len < max_length:
                    neg_log_likelihood = outputs.loss * trg_len
                    lm_logits = outputs.logits[..., -trg_len-1:-1, :]
                    labels = target_ids[..., -trg_len:]
                else:
                    neg_log_likelihood = outputs.loss * (max_length - 1)
                    lm_logits = outputs.logits[..., :-1, :]
                    labels = target_ids[..., 1:]
                neg_log_likelihood = neg_log_likelihood.to(torch.float32).squeeze().cpu()
                lm_logits = lm_logits.to(torch.float32)

                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)).cpu()
                token_ppls = loss.tolist()
                tokens_to_predict = labels.view(-1).cpu().tolist()

        nlls.append(neg_log_likelihood)
        all_token_ppls.append(token_ppls)
        all_tokens_to_predict.append(tokens_to_predict)
        all_num_prepended_token.append(num_prepended_token)
        if per_doc_ranking_loss:
            all_per_doc_ranking_loss.append(per_doc_ranking_loss)
        assert len(all_token_ppls) == len(all_tokens_to_predict)

        prev_end_loc = end_loc
        idx += 1
        if end_loc == dataset_len:
            break

    assert retrieval_dataset is None or len(retrieval_dataset) == idx


    print("Input length: mean {}, sum {}", np.mean(all_num_prepended_token),
          np.sum(all_num_prepended_token))
    ppl = torch.exp(torch.stack(nlls).sum() / counter).item()
    print("Perplexity:", ppl)
    ppl_to_assert = np.exp(sum([sum(x) for x in all_token_ppls]) / counter)
    assert np.abs(ppl - ppl_to_assert) < 1e-3, f"{ppl:.3f}, {ppl_to_assert:.3f}"

    if output_dir is not None:
        d = {"eval_perplexity": ppl}
        if retrieval_dataset is not None:
            d["num_input_no_retrieval"] = num_inputs_no_retrieval
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")

        with open(os.path.join(output_dir, "ppls.pkl"), "wb") as f:
            to_dump = (all_token_ppls, all_tokens_to_predict, all_chosen_doc_ids, all_per_doc_ranking_loss) if all_chosen_doc_ids \
                else (all_token_ppls, all_tokens_to_predict, all_per_doc_ranking_loss)
            pickle.dump(to_dump, f)


def main(args):
    if args.output_dir is not None:
        os.makedirs(args.output_dir)
    print_args(args, output_dir=args.output_dir)
    device = "cuda:{}".format(args.first_gpu) if torch.cuda.is_available() else "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()
    print("device count: {}".format(device_count))
    data_parallel = device_count > 1 and not args.model_parallelism and args.retrieved_file is not None and \
                    args.ranking_strategy in ["logprob", "oracle"]

    config = AutoConfig.from_pretrained(args.model_name)
    model_args = {
        "cache_dir": args.cache_dir
    }
    if args.model_parallelism:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_args).eval()
    if not args.model_parallelism:
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Model context size (e.g., 1024 for GPT-2)
    max_length = args.max_length
    model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings
    if max_length is None or max_length > model_max_length:
        max_length = model_max_length

    if data_parallel:
        model = torch.nn.DataParallel(model)

    if args.load_from == "hf":
        dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split)
        dataset = "".join([x["text"] if x["text"] else " \n" for x in dataset])
    else:
        with open(args.dataset_path, "r") as f:
            dataset = f.read()

    transformers.logging.set_verbosity_error()
    retrieval_dataset = None
    if args.retrieved_file is not None:
        with open(args.retrieved_file, "r") as f:
            retrieval_dataset = json.load(f)

    t1 = time.time()
    eval_dataset(
        model,
        tokenizer,
        dataset,
        device,
        max_length=max_length,
        output_dir=args.output_dir,
        stride=args.stride,
        normalization_level=args.normalization_level,
        retrieval_dataset=retrieval_dataset,
        retrieval_max_length=args.retrieved_max_length,
        ranking_strategy=args.ranking_strategy,
        num_docs_to_rank=args.num_docs_to_rank,
        num_tokens_to_rank_logprob=args.ranking_logprob_past_tokens,
        compression_method=args.compression_method,
        top_k=args.top_k,
    )

    t2 = time.time()
    print("Total time: {} seconds".format(int(t2 - t1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--first_gpu", type=int, default=0)

    # Dataset params
    parser.add_argument("--load_from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--normalization_level", choices=["word", "token"], default="word")

    # retrieval params
    parser.add_argument("--retrieved_file", type=str, default=None)
    parser.add_argument("--retrieved_max_length", type=int, default=256)
    parser.add_argument("--ranking_strategy", type=str, choices=["first", "logprob", "oracle", "random"], default="first")
    parser.add_argument("--num_docs_to_rank", type=int, default=-1)
    parser.add_argument("--ranking_logprob_past_tokens", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=1)

    # compression params
    parser.add_argument("--compression_method", type=str, choices=["none", "bow", "ner"], default="none")

    args = parser.parse_args()

    main(args)
