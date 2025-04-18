# https://github.com/NoviScl/GPT3-Reliability/blob/f2705721a0a921003d30bb9fe7eed14a28a0bab2/utils.py
import re
import string
import collections
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
stops = set(stopwords.words('english'))
puncs = list(string.punctuation)

def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(puncs)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    # print(a_gold, a_pred)
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def single_ans_em(pred, gold):
    # pred: prediction string
    # gold: a list of gold answer strings
    if type(gold) !=list:
        gold = [gold]
    pred = answer_extract_textqa(pred)
    return max(compute_exact(pred, a) for a in gold)

def single_ans_f1(pred, gold):
    # pred: prediction string
    # gold: a list of gold answer strings
    if type(gold) !=list:
        gold = [gold]
    pred = answer_extract_textqa(pred)
    return max(compute_f1(pred, a) for a in gold)

def get_exact_match(answers1, answers2):
    if type(answers1)==list:
        if len(answers1)==0:
            return 0
        return np.max([get_exact_match(a, answers2) for a in answers1])
    if type(answers2)==list:
        if len(answers2)==0:
            return 0
        return np.max([get_exact_match(answers1, a) for a in answers2])
    return (normalize_answer(answers1) == normalize_answer(answers2))


def get_f1(answers, predictions, is_equal=get_exact_match):
    '''
    :answers: a list of list of strings
    :predictions: a list of strings
    '''
    assert len(answers)>0 and len(predictions)>0, (answers, predictions)
    occupied_answers = [False for _ in answers]
    occupied_predictions = [False for _ in predictions]
    for i, answer in enumerate(answers):
        for j, prediction in enumerate(predictions):
            if occupied_answers[i] or occupied_predictions[j]:
                continue
            em = is_equal(answer, prediction)
            if em:
                occupied_answers[i] = True
                occupied_predictions[j] = True
    assert np.sum(occupied_answers)==np.sum(occupied_predictions)
    a, b = np.mean(occupied_answers), np.mean(occupied_predictions)
    if a+b==0:
        return 0
    return 2*a*b/(a+b)

def answer_match_textqa(pred, ans):
    pred = answer_extract_textqa(pred)
    return normalize_answer(pred) == normalize_answer(ans)

def answer_extract_textqa(pred):
    prefix = "answer is "
    if prefix in pred:
        idx = pred.rfind(prefix)
        # print ("extracted ans string: ", pred[idx + len(prefix) : ])
        return pred[idx + len(prefix) : ]
    return pred.strip()