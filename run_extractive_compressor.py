"""Implement baseline selectors"""
import pandas as pd
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModel,DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer, DPRContextEncoder
from tqdm import tqdm

from rank_bm25 import BM25Okapi

def get_bm25_scores(data_row, top_k):
    if len(data_row['retrieved_docs']) == 0:
        return []
    if top_k == -1:
        corpus = [data['text'] for data in data_row['retrieved_docs']]
    else:
        corpus = [data['text'] for data in data_row['retrieved_docs'][:top_k]]

    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    query = data_row['query']
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    return doc_scores

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_contriever_scores(model, tokenizer, data_row, device, top_k):
    if len(data_row['retrieved_docs']) == 0:
        return []
    if top_k == -1:
        corpus = [data['text'] for data in data_row['retrieved_docs']]
    else:
        corpus = [data['text'] for data in data_row['retrieved_docs'][:top_k]]
    query = data_row['query']
    inputs = tokenizer([query] + corpus, padding=True, truncation=True, return_tensors='pt').to(device)
    # Compute token embeddings
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
    scores = []
    for i in range(len(corpus)):
        scores.append((embeddings[0] @ embeddings[i+1]).item())
    return scores

def get_dpr_scores(q_model, q_tokenizer, c_model, c_tokenizer, data_row, device, top_k):
    if len(data_row['retrieved_docs']) == 0:
        return []
    if top_k == -1:
        corpus = [data['text'] for data in data_row['retrieved_docs']]
    else:
        corpus = [data['text'] for data in data_row['retrieved_docs'][:top_k]]
    query = data_row['query']
    q_input = q_tokenizer(query, return_tensors='pt')["input_ids"].to(device)
    q_embedding = q_model(q_input).pooler_output.detach().cpu()
    doc_input = c_tokenizer(corpus, padding=True, truncation=True, return_tensors='pt').to(device)
    doc_embeddings = c_model(**doc_input).pooler_output.detach().cpu()
    scores = []
    for i in range(len(corpus)):
        scores.append((q_embedding @ doc_embeddings[i]).item())
    return scores


def main():
    argparse = ArgumentParser()
    argparse.add_argument("--input_data", dest='input_data', required=True)
    argparse.add_argument("--model_type", dest='model_type', required=False,
                          choices=['dpr', 'bm25', 'facebook/contriever-msmarco', 'facebook/contriever'])
    argparse.add_argument("--model_path", dest='model_path', required=False)
    argparse.add_argument("--output_file", dest='output_file', type=str, required=True)
    argparse.add_argument("--device", dest='device', default=0, type=int)
    argparse.add_argument("--top_k", dest='top_k', default=30, type=int)

    args = argparse.parse_args()
    print(args)


    input_data_df = pd.read_json(args.input_data)
    print(input_data_df.columns)


    if args.model_path:
        # trained compressor
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModel.from_pretrained(args.model_path).to('cuda:{}'.format(args.device))

        # get contriever scores
        contriever_scores = []

        for _, data in tqdm(input_data_df.iterrows(), total=len(input_data_df)):
            scores = get_contriever_scores(model, tokenizer, data, 'cuda:{}'.format(args.device), top_k=args.top_k)
            contriever_scores.append(scores)

        input_data_df['compressor_scores'] = contriever_scores

    else:
        # run baseline compressors
        if 'contriever' in args.model_type:
            tokenizer = AutoTokenizer.from_pretrained(args.model_type)
            model = AutoModel.from_pretrained(args.model_type).to('cuda:{}'.format(args.device))

            # get contriever scores
            contriever_scores = []

            for _, data in tqdm(input_data_df.iterrows(), total=len(input_data_df)):
                scores = get_contriever_scores(model, tokenizer, data, 'cuda:{}'.format(args.device), top_k=args.top_k)
                contriever_scores.append(scores)


            input_data_df['contriever_scores'] = contriever_scores
        elif args.model_type == 'dpr':
            q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            q_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to('cuda:{}'.format(args.device))
            c_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            c_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to('cuda:{}'.format(args.device))

            dpr_scores = []
            for _, data in tqdm(input_data_df.iterrows(), total=len(input_data_df)):
                scores = get_dpr_scores(q_model, q_tokenizer, c_model, c_tokenizer, data, 'cuda:{}'.format(args.device), top_k=args.top_k)
                dpr_scores.append(scores)
            input_data_df['dpr_scores'] = dpr_scores
        elif args.model_type == 'bm25':
            input_data_df['bm25_scores'] = input_data_df.apply(lambda data: get_bm25_scores(data, top_k=args.top_k), axis=1)
        else:
            raise ValueError("Incorrect ranking model")

    input_data_df.to_json(args.output_file,
                                 orient='records')

if __name__ == "__main__":
    main()