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
from fastNLP import Trainer
from peng.model.metrics import Seq2SeqSpanMetric
from peng.model.losses import Seq2SeqLoss
from peng.model.prompt_auxilary_loss import PromptAuxSeq2SeqLoss
from torch import optim
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback
from fastNLP import FitlogCallback
from fastNLP.core.sampler import SortedSampler
from peng.model.generator import SequenceGeneratorModel
from peng.model.prompt_generator import SequenceGeneratorPromptModel
import fitlog
import random
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='pengb/14lap', type=str)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)
parser.add_argument('--save_model', type=str, default=None)
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--fewshot', action='store_true')
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=12345)

args= parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.log_dir, exist_ok=True)
# fitlog.debug()
fitlog.set_log_dir(args.log_dir)

lr = args.lr
n_epochs = args.n_epochs
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
save_model = args.save_model
fitlog.add_hyper(args)

#######hyper
#######hyper

cache_dir = f"prompt_{args.prompt}_caches" if args.prompt is not None else "caches"
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
        if args.prompt is not None:
            pipe = BartBPEABSAPromptPipe(tokenizer=bart_name, opinion_first=opinion_first, ptype=args.prompt)
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
    'lcx/16res': 1.2,
}[dataset_name]

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

bos_token_id = 0  #
eos_token_id = 1  #
label_ids = list(mapping2id.values())
print(args.prompt)
if args.prompt:
    model = BartPromptSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, config=args, decoder_type=decoder_type,
                                     copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False)
else:
    model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                     copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False)
vocab_size = len(tokenizer)
print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))

if args.prompt:
    model = SequenceGeneratorPromptModel(model, bos_token_id=bos_token_id,
                                eos_token_id=eos_token_id,
                                max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                                repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                                restricter=None)
else:
    model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                                eos_token_id=eos_token_id,
                                max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                                repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                                restricter=None)

import torch
if torch.cuda.is_available():
    # device = list([i for i in range(torch.cuda.device_count())])
    device = 'cuda'
else:
    device = 'cpu'
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

parameters = []
params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = [param for name, param in model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name)]
parameters.append(params)

params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr':lr, 'weight_decay':0}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

optimizer = optim.AdamW(parameters)


callbacks = []
callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
callbacks.append(FitlogCallback(data_bundle.get_dataset('test'), log_loss_every=1))

sampler = None
# sampler = ConstTokenNumSampler('src_seq_len', max_token=1000)
sampler = BucketSampler(seq_len_field_name='src_seq_len')
metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), opinion_first=opinion_first)


model_path = None
if save_model:
    model_path = args.save_model

trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                  loss= PromptAuxSeq2SeqLoss() if args.prompt else Seq2SeqLoss(),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                  num_workers=2, n_epochs=n_epochs, print_every=1,
                  dev_data=data_bundle.get_dataset('dev'), metrics=metric, metric_key='triple_f',
                  validate_every=-1, save_path=model_path, use_tqdm=True, device=device,
                  callbacks=callbacks, check_code_level=0, test_use_tqdm=False,
                  test_sampler=SortedSampler('src_seq_len'), dev_batch_size=batch_size)

trainer.train(load_best_model=False)
