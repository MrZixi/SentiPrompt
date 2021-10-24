from fastNLP.io import Pipe, DataBundle, Loader
import os
import json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer
import numpy as np
from itertools import chain
from functools import cmp_to_key
import random
from copy import deepcopy
from itertools import product

ASPECT_PROMPT_TYPE = {'type1': [["the"], ["is"], ["?"]],
                      'type2': [["<unk0>", "<unk1>", "<unk2>"], ["<unk3>", "<unk4>", "<unk5>"], ["?"]],
                      'type3': [["<unk0>", "<unk2>"], ["<unk4>", "<unk5>"], ["?"]],
                      'type4': [["<unk0>"], ["<unk5>"], ["?"]]}
OPINION_PROMPT_TYPE = {'type1': ["this", "is"],
                       'type2': ["<unk6>", "<unk7>"],
                       'type3': ["<unk6>", "<unk7>"],
                       'type4': ["<unk6>", "<unk7>"]}
# ASPECT_SENTIMENT_PROMPT = [["the"], ["is"], ["?"]]
# OPINION_PROMPT = ["this", "is"]
# ASPECT_SENTIMENT_PROMPT = [["<unk0>", "<unk1>", "<unk2>"], ["<unk3>", "<unk4>", "<unk5>"], ["?"]]

# ASPECT_SENTIMENT_PROMPT = [["<unk0>", "<unk2>"], ["<unk4>", "<unk5>"], ["?"]]
# OPINION_PROMPT = ["<unk6>", "<unk7>"]

def cmp_aspect(v1, v2):
    if v1[0]['from']==v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']

def cmp_opinion(v1, v2):
    if v1[1]['from']==v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']


class BartBPEABSAPromptPipe(Pipe):
    def __init__(self, tokenizer='facebook/bart-base', opinion_first=True, ptype='type1'):
        super(BartBPEABSAPromptPipe, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mapping = {  # so that the label word can be initialized in a better embedding.
            'POS': '<<positive>>',
            'NEG': '<<negative>>',
            'NEU': '<<neutral>>'
        }
        self.opinion_first = opinion_first  # 是否先生成opinion

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens

        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}
        self.ptype = ptype

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def construct_prompt(self, aspect_opinions, gt, src_token_bpes, src_tokens):
        prompt_pos, prompt_tgt, prompt_mask_pos = [], [], []
        temp_tokens = deepcopy(src_token_bpes)

        # aspects, opinions = aspect_opinions
        temp_s, temp_e = sum(list(map(len, temp_tokens))), None

        for aspects, opinions in aspect_opinions:
            for word in ASPECT_PROMPT_TYPE[self.ptype][0]:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            temp_e = sum(list(map(len, temp_tokens)))
            prompt_pos.extend([x for x in range(temp_s, temp_e)])

            for word in aspects['term']:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)

            temp_s = sum(list(map(len, temp_tokens)))
            for word in ASPECT_PROMPT_TYPE[self.ptype][1]:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            temp_e = sum(list(map(len, temp_tokens)))
            prompt_pos.extend([x for x in range(temp_s, temp_e)])

            for word in opinions['term']:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            for word in ASPECT_PROMPT_TYPE[self.ptype][2]:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            prompt_mask_pos.append(sum(list(map(len, temp_tokens))))
            temp_tokens.append([self.tokenizer.mask_token_id])

            if aspects['index'] == opinions['index']:
                prompt_tgt.append(self.tokenizer.convert_tokens_to_ids(["yes"])[0])
                prompt_tgt.append(self.mapping2targetid[aspects['polarity']]+2)
                temp_s = sum(list(map(len, temp_tokens)))
                for word in OPINION_PROMPT_TYPE[self.ptype]:
                    bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                    bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                    temp_tokens.append(bpes)
                temp_e = sum(list(map(len, temp_tokens)))
                prompt_pos.extend([x for x in range(temp_s, temp_e)])
                prompt_mask_pos.append(sum(list(map(len, temp_tokens))))
                temp_tokens.append([self.tokenizer.mask_token_id])
            else:
                prompt_tgt.append(self.tokenizer.convert_tokens_to_ids(["no"])[0])

        for aspects, opinions in gt:
            if len(aspects['term']) == 1:
                aspects['term'] = src_tokens[random.randint(max(0, aspects['from'] - 3), max(0, aspects['from'] - 1)):random.randint(min(aspects['to'] + 1, len(src_tokens)), min(aspects['to'] + 3, len(src_tokens)))]
            else:
                aspects['term'] = aspects['term'][:random.randint(0, len(aspects['term']) - 1) + 1]
            temp_s = sum(list(map(len, temp_tokens)))
            for word in ASPECT_PROMPT_TYPE[self.ptype][0]:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            temp_e = sum(list(map(len, temp_tokens)))
            prompt_pos.extend([x for x in range(temp_s, temp_e)])

            for word in aspects['term']:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)

            temp_s = sum(list(map(len, temp_tokens)))
            for word in ASPECT_PROMPT_TYPE[self.ptype][1]:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            temp_e = sum(list(map(len, temp_tokens)))
            prompt_pos.extend([x for x in range(temp_s, temp_e)])

            for word in opinions['term']:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            for word in ASPECT_PROMPT_TYPE[self.ptype][2]:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            prompt_mask_pos.append(sum(list(map(len, temp_tokens))))
            temp_tokens.append([self.tokenizer.mask_token_id])
            prompt_tgt.append(self.tokenizer.convert_tokens_to_ids(["no"])[0])

            if len(opinions['term']) == 1:
                opinions['term'] = src_tokens[random.randint(max(0, opinions['from'] - 3), max(0, opinions['from'] - 1)):random.randint(min(opinions['to'] + 1, len(src_tokens)), min(opinions['to'] + 3, len(src_tokens)))]
            else:
                opinions['term'] = opinions['term'][:random.randint(0, len(opinions['term']) - 1) + 1]
            temp_s = sum(list(map(len, temp_tokens)))
            for word in ASPECT_PROMPT_TYPE[self.ptype][0]:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            temp_e = sum(list(map(len, temp_tokens)))
            prompt_pos.extend([x for x in range(temp_s, temp_e)])

            for word in aspects['term']:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)

            temp_s = sum(list(map(len, temp_tokens)))
            for word in ASPECT_PROMPT_TYPE[self.ptype][1]:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            temp_e = sum(list(map(len, temp_tokens)))
            prompt_pos.extend([x for x in range(temp_s, temp_e)])

            for word in opinions['term']:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            for word in ASPECT_PROMPT_TYPE[self.ptype][2]:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                temp_tokens.append(bpes)
            prompt_mask_pos.append(sum(list(map(len, temp_tokens))))
            temp_tokens.append([self.tokenizer.mask_token_id])
            prompt_tgt.append(self.tokenizer.convert_tokens_to_ids(["no"])[0])

        temp_tokens.append([self.tokenizer.eos_token_id])
        return prompt_pos, list(chain(*temp_tokens)), prompt_tgt, prompt_mask_pos

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        words: List[str]
        aspects: [{
            'index': int
            'from': int
            'to': int
            'polarity': str
            'term': List[str]
        }],
        opinions: [{
            'index': int
            'from': int
            'to': int
            'term': List[str]
        }]

        输出为[o_s, o_e, a_s, a_e, c]或者[a_s, a_e, o_s, o_e, c]
        :param data_bundle:
        :return:
        """
        target_shift = len(self.mapping) + 2  # 是由于第一位是sos，紧接着是eos, 然后是

        def prepare_target(ins):
            prompt_pos = []
            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]

            for word in raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.tokenizer.sep_token_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            target = [0]  # 特殊的开始
            target_spans = []
            _word_bpes = list(chain(*word_bpes))

            aspects_opinions = [(a, o) for a, o in zip(ins['aspects'], ins['opinions'])]
            if self.opinion_first:
                aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_opinion))
            else:
                aspects_opinions = sorted(aspects_opinions, key=cmp_to_key(cmp_aspect))

            for aspects, opinions in aspects_opinions:  # 预测bpe的start
                if not aspects['index'] ==  opinions['index']:
                    print(raw_words)
                    print(aspects['index'])
                    print(opinions['index'])
                assert aspects['index'] == opinions['index']
                a_start_bpe = cum_lens[aspects['from']]  # 因为有一个sos shift
                a_end_bpe = cum_lens[aspects['to']-1]  # 这里由于之前是开区间，刚好取到最后一个word的开头
                o_start_bpe = cum_lens[opinions['from']]  # 因为有一个sos shift
                o_end_bpe = cum_lens[opinions['to']-1]  # 因为有一个sos shift
                # 这里需要evaluate是否是对齐的
                for idx, word in zip((o_start_bpe, o_end_bpe, a_start_bpe, a_end_bpe),
                                     (opinions['term'][0], opinions['term'][-1], aspects['term'][0], aspects['term'][-1])):                 
                    if not (_word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]):
                        print(raw_words)
                        print(word)
                    assert _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                           _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]

                if self.opinion_first:
                    target_spans.append([o_start_bpe+target_shift, o_end_bpe+target_shift,
                                         a_start_bpe+target_shift, a_end_bpe+target_shift])
                else:
                    target_spans.append([a_start_bpe+target_shift, a_end_bpe+target_shift,
                                         o_start_bpe+target_shift, o_end_bpe+target_shift])
                target_spans[-1].append(self.mapping2targetid[aspects['polarity']]+2)   # 前面有sos和eos
                target_spans[-1] = tuple(target_spans[-1])
            target.extend(list(chain(*target_spans)))
            target.append(1)  # append 1是由于特殊的eos
            
            random_pairs = random.sample(list(product(ins['aspects'], ins['opinions'])), len(ins['aspects']))
            prompt_pos, prompt_src, prompt_tgt, prompt_mask_pos = self.construct_prompt(aspect_opinions=random_pairs,
                                                                                        gt=zip(ins['aspects'], ins['opinions']),
                                                                                        src_token_bpes=word_bpes, src_tokens=raw_words)
            return {'prompt_pos': prompt_pos, 'prompt_src':prompt_src, 'prompt_tgt':prompt_tgt, 
                    'prompt_mask_pos': prompt_mask_pos, 'tgt_tokens': target, 'target_span': target_spans, 
                    'src_tokens': list(chain(*word_bpes))}

        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='Pre. tgt.')

        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'prompt_src', 'prompt_pos', 'prompt_mask_pos')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'prompt_tgt')
        
        return data_bundle

    def process_from_file(self, paths, demo=False, fewshot=False) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = ABSALoader(demo=demo, fewshot=fewshot).load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class ABSALoader(Loader):
    def __init__(self, demo=False, fewshot=False):
        super().__init__()
        self.demo = demo
        self.fewshot = fewshot

    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ds = DataSet()
        if self.fewshot and "train" in path:
            data = random.sample(data, int(0.1 * len(data)))
        for ins in data:
            tokens = ins['words']
            aspects = ins['aspects']
            opinions = ins['opinions']
            # print(ins)
            if not len(aspects)==len(opinions):
                print(path)
                print(ins)    
            assert len(aspects)==len(opinions)
            if len(aspects) == 1:
                # print(aspects)
                continue
            ins = Instance(raw_words=tokens, aspects=aspects, opinions=opinions)
            ds.append(ins)
            if self.demo and len(ds)>30:
                break
        print(path)
        print(sum([len(ins['aspects']) for ins in data]))
        return ds


if __name__ == '__main__':
    data_bundle = BartBPEABSAPromptPipe().process_from_file('/root/code/BARTABSA/data/lcx/16res')
    print(data_bundle)

