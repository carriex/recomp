import pandas as pd
import eval_utils
from argparse import ArgumentParser
import ast

parser = ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
args = parser.parse_args()


input_df = pd.read_csv(args.input_file)
input_df['em'] = input_df.apply(lambda data: eval_utils.single_ans_em(gold=ast.literal_eval(data['gold_answers']),
                                                                         pred=data['pred_answer']), axis=1)
input_df['f1'] = input_df.apply(lambda data: eval_utils.single_ans_f1(gold=ast.literal_eval(data['gold_answers']),
                                                                         pred=data['pred_answer']), axis=1)

print(input_df[['em', 'f1']].mean())