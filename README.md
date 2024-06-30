# recomp

This is the repository for the paper [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/pdf/2310.04408.pdf).

## Data
Download the files from [here](https://drive.google.com/drive/folders/1X-BHlZ_HG8tRL-7u70TZGKYZ3W14fnJn?usp=sharing) and place them in the `data/` directory.
```
data/
- prompts/ # prompts with uncompressed and compressed retrieved documents
- extractive_compressor_inputs/ # sentences passed to extractive compressors
- abstractive_compressor_inputs/ # retrieved documents passed to abstractive compressors
```

## Models
Extractive compressor
* [nq_extractive](https://huggingface.co/fangyuan/nq_extractive_compressor)
* [tqa_extractive](https://huggingface.co/fangyuan/tqa_extractive_compressor)
* [hotpotqa_extractive](https://huggingface.co/fangyuan/hotpotqa_extractive_compressor)

Abstractive compressor
* [nq_abstractive](https://huggingface.co/fangyuan/nq_abstractive_compressor)
* [tqa_abstractive](https://huggingface.co/fangyuan/tqa_abstractive_compressor)
* [hotpotqa_abstractive](https://huggingface.co/fangyuan/hotpotqa_abstractive)

## Language modelling task

### Preparing retrieved documents 

Follow steps [here](https://github.com/AI21Labs/in-context-ralm/tree/main?tab=readme-ov-file#retrieval) to prepare retrieval documents.

We use the below command (with this [script](https://github.com/AI21Labs/in-context-ralm/blob/main/prepare_retrieval_data.py)).

```bash
python prepare_retrieval_data.py \
--retrieval_type sparse \
--tokenizer_name gpt2 \
--max_length 1024 \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split validation \
--index_name wikipedia-dpr \
--forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
--stride 32 \
--output_file gpt2_wikitext_validation_retrieval_files_32 \
--num_tokens_for_query 32 \
--num_docs 16
```

### Evaluation

Run the below command to evaluate perplexity with different sets of retrieved documents. For example, to evaluate uncompressed RALM with GPT-2 on wikitext, run:

```bash
python eval_lm.py \
--model_name gpt-2 \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split validation \
--output_dir outputs/gpt-2 \
--stride 32 \
--max_length 1024 \
--model_parallelism \
--retrieved_file gpt2_wikitext_validation_retrieval_files_32
```


## QA tasks

Run the below script to prompt FLAN-UL2. We release compressed documents in `data/` for each dataset.

```bash
python prompt_flan.py \
--input_data_csv_file [input_data_file] \
--output_data_csv_file [name_of_output_data]
```

## Extractive compressor

Run the below script to score the sentences with the train compressor by passing the path to `--model_path`. To run baseline compressors, pass in `--model_type` instead.
For example, to score sentences with extractive compressor for top 5 retrieved documents from NQ, run:

```bash
python run_extractive_compresor.py \
  --input_data data/extractive_compressor_intputs/flan_ul2_nq_5shot_top_5_passage_new_msmarco_sent.json \
  --model_path  fangyuan/nq_extractive_compressor \
  --output_file outputs/flan_ul2_nq_5shot_top_5_passage_msmarco_sent_with_scores.json \
  --top_k -1 # consider all sentences
```

## Abstractive compressor 

Run the below script to compress retrieved documents with abstractive compressor. For example, to compress top 5 retrieved documents from NQ, run:

```bash
python train_hf_summarization_model.py \
--model_name_or_path fangyuan/nq_abstractive_compressor \
--do_predict \
--test_file  abstractive_compressor_inputs/nq_dev_contriever_msmarco_top_5_docs.json \
--max_target_length 512 \
--output_dir outputs/ \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=16 \
--predict_with_generate
```


## Training

### Extractive compressor
Download the pre-processed training data [here](https://drive.google.com/drive/folders/1Roahn6qQxB_zZ5j4ZtNm4GQk68m63nqn?usp=sharing)

Run the below script to train abstractive compressor for (e.g. NQ):

```bash
python train_extractive_compressor.py \
--model_name facebook/contriever \
--train_data_path data/extractive_training/nq/train.json \
--dev_data_path data/extractive_training/nq/dev.json
```

### Abstractive compressor
Download the pre-processed training data [here](https://drive.google.com/drive/folders/1Roahn6qQxB_zZ5j4ZtNm4GQk68m63nqn?usp=sharing)

Run the below script to train abstractive compressor for (e.g. NQ):

```bash
python train_hf_summarization_model.py \
    --model_name_or_path t5-large \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --train_file data/abstractive_training/nq/train.json \
    --validation_file data/abstractive_training/nq/dev.json \
    --report_to tensorboard \
    --max_target_length 512 \
    --output_dir [output_dir] \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=2 \
    --per_device_eval_batch_size=8 \
    --overwrite_output_dir \
    --predict_with_generate \
    --save_total_limit 3 \
    --logging_first_step True \
    --max_eval_samples 10000 \
    --load_best_model_at_end
```