# script to prompt FLAN-UL2
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import pandas as pd
import ast
from tqdm import tqdm
import numpy as np
import itertools
from argparse import ArgumentParser


def main():
    argparse = ArgumentParser()
    argparse.add_argument("--input_data_csv_file", dest='input_data_csv_file', required=True)
    argparse.add_argument("--output_data_csv_file", dest='output_data_csv_file', required=True)
    argparse.add_argument("--batch_size", dest='bs', default=1, type=int)

    args = argparse.parse_args()
    print(args)

    model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

    print(">> loaded the model")
    input_df = pd.read_csv(args.input_data_csv_file)
    prompts = list(input_df['prompt'])

    generated_answers = []

    with torch.no_grad():
        for prompt in tqdm(prompts):

            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(inputs, max_length=200)

            output = tokenizer.decode(outputs[0])
            generated_answers.append(output)

        assert len(generated_answers) == len(prompts), "{} processed data, {} data".format(len(generated_answers),
                                                                                           len(prompts))

    input_df['generated_answers'] = generated_answers
    input_df.to_csv('data/completions/flan_ul2_{}'.format(args.output_data_csv_file),
                        index=False)

if __name__ == "__main__":
    main()