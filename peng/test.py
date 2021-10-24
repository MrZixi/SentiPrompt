import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
warnings.filterwarnings('ignore')
from peng.data.pipe import BartBPEABSAPipe
from peng.data.prompt_pipe import BartBPEABSAPromptPipe
from peng.model.bart_absa import BartSeq2SeqModel
from peng.model.bart_prompt_absa import BartPromptSeq2SeqModel
from fastNLP import Tester
from peng.model.metrics import Seq2SeqSpanMetric
from fastNLP import GradientClipCallback, cache_results, WarmupCallback
from peng.model.generator import SequenceGeneratorModel
from peng.model.prompt_generator import SequenceGeneratorPromptModel
import fitlog

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='pengb/14lap', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--prompt', action='store_true')
parser.add_argument('--fewshot', action='store_true')
parser.add_argument('--hidden_size', type=int, default=1024)

args = parser.parse_args()

os.makedirs(args.log_dir, exist_ok=True)
# fitlog.debug()
fitlog.set_log_dir(args.log_dir)

batch_size = args.batch_size
num_beams = args.num_beams
dataset_name = args.dataset_name
opinion_first = args.opinion_first
length_penalty = args.length_penalty
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
bart_name = args.bart_name
use_encoder_mlp = args.use_encoder_mlp
fitlog.add_hyper(args)

#######hyper
#######hyper

cache_dir = "prompt_caches" if args.prompt else "caches"
if args.fewshot:
    cache_dir = "fewshot_" + cache_dir
demo = False
if demo:
    cache_fn = f"{cache_dir}/data_{bart_name}_{dataset_name}_{opinion_first}_demo.pt"
else:
    cache_fn = f"{cache_dir}/data_{bart_name}_{dataset_name}_{opinion_first}.pt"

if args.fewshot:
    @cache_results(cache_fn, _refresh=True)
    def get_data():
        if args.prompt:
            pipe = BartBPEABSAPromptPipe(tokenizer=bart_name, opinion_first=opinion_first)
        else:
            pipe = BartBPEABSAPipe(tokenizer=bart_name, opinion_first=opinion_first)
        data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=demo, fewshot=args.fewshot)
        return data_bundle, pipe.tokenizer, pipe.mapping2id
else:
    @cache_results(cache_fn, _refresh=False)
    def get_data():
        if args.prompt:
            pipe = BartBPEABSAPromptPipe(tokenizer=bart_name, opinion_first=opinion_first)
        else:
            pipe = BartBPEABSAPipe(tokenizer=bart_name, opinion_first=opinion_first)
        data_bundle = pipe.process_from_file(f'../data/{dataset_name}', demo=demo, fewshot=args.fewshot)
        return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()
max_len = 10
max_len_a = {
    'penga/14lap': 0.9,
    'penga/14res': 1,
    'penga/15res': 1.2,
    'penga/16res': 0.9,
    'pengb/14lap': 1.1,
    'pengb/14res': 1.2,
    'pengb/15res': 0.9,
    'pengb/16res': 1.2,
    'lcx/14lap': 1.1,
    'lcx/14res': 1.2,
    'lcx/15res': 0.9,
    'lcx/16res': 1.2
}[dataset_name]

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

bos_token_id = 0  #
eos_token_id = 1  #
label_ids = list(mapping2id.values())

import torch

if args.model_path == '':
    _first = os.listdir(os.path.join(args.log_dir, 'models'))[0]
    args.model_path = os.path.join(args.log_dir, os.path.join('models', _first))

model = torch.load(args.model_path)

if torch.cuda.is_available():
    # device = list([i for i in range(torch.cuda.device_count())])
    device = 'cuda'
else:
    device = 'cpu'

model.to(device)


callbacks = []

metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), opinion_first=opinion_first, output_bad_cases=args.log_dir)

tester = Tester(data=data_bundle.get_dataset('test'), model=model,
                metrics=metric, batch_size=batch_size, num_workers=2)
tester.test()