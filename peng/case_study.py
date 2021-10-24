import os
import json
import argparse
import ast
from transformers import AutoTokenizer
POLARITY = ['POS', 'NEG', 'NEU']
def process_pred(pred, tokens):
    pred_terms = []
    pred = ast.literal_eval(pred)
    pred = [pred[i:i + 5] for i in range(0, len(pred), 5)]
    for triplet_span in pred:
        # print(triplet_span)
        if len(triplet_span) < 5 or triplet_span[4] > 5 or triplet_span[0] > triplet_span[1] or triplet_span[2] > triplet_span[3]:
            print('Error prediction')
            # print(triplet_span)
            continue
        temp_pol = POLARITY[triplet_span[4] - 2]
        temp_asp = tokens[triplet_span[0] - 6:triplet_span[1] - 5]
        temp_op = tokens[triplet_span[2] - 6:triplet_span[3] - 5]
        pred_terms.append((" ".join(temp_asp).replace('\u0120', ''), " ".join(temp_op).replace('\u0120', ''), temp_pol))
    return pred_terms

def load(args):
    with open(os.path.join(args.data_path, 'test_convert.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(args.pred_path, 'r',  encoding='utf-8') as f:
        preds = f.readlines()
    with open(args.pred_path.replace('pred', 'tgt'), 'r',  encoding='utf-8') as f:
        tgts = f.readlines()
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
    with open(os.path.join(args.output_dir, 'case.log'), 'w', encoding='utf-8') as f:
        print(len(preds))
        print(len(data))
        print(len(tgts))
        for index, (ins, pred, tgt) in enumerate(zip(data, preds, tgts)):
            tgt = ast.literal_eval(tgt)
            temp_tgt = []
            for x in tgt:
                temp_tgt.extend(x)
            tgt = temp_tgt
            tokens = ins['words']
            aspects = ins['aspects']
            opinions = ins['opinions']
            spans = process_pred(pred, tokenizer.tokenize(ins['raw_words']))
            pred = ast.literal_eval(pred)
            # flag = True
            # for aspect_gt, opinion_gt, (aspect_pred, opinion_pred, pol) in zip(aspects, opinions, spans):
            #     if not "".join(aspect_gt['term']) in aspect_pred.replace(' ', '') or not "".join(opinion_gt['term']) in opinion_pred.replace(' ', '') or not pol == aspect_gt['polarity']:
            #         flag = False
            
            # if flag:
            #     continue
            if pred == tgt:
                continue
            f.write(str(index))
            f.write('\n')
            f.write(" ".join(tokens))
            f.write('\n')
            for aspect, opinion in zip(aspects, opinions):
                f.write(" ".join(aspect['term']) + '/' + " ".join(opinion['term']) + '/' + aspect['polarity'] + '| ')
            f.write('\n')
            for aspect, opinion, polarity in spans:
                f.write(aspect + '/' + opinion + '/' + polarity + '| ')
            f.write('\n')
            f.write('\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--pred_path', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    load(args)