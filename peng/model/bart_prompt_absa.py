import torch
from .modeling_bart import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import nn
from peng.model.bart_absa import FBartDecoder, CaGFBartDecoder, BartState
from peng.model.prompt_encoder import PromptEncoder
import math
from copy import deepcopy


class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len=None, src_embs=None):

        if src_seq_len is not None:
            mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        else:
            mask = None
        if src_embs is not None:
            dict = self.bart_encoder(input_ids=src_tokens,
                                     return_dict=True,
                                     output_hidden_states=True,
                                     input_embs=src_embs)
            encoder_outputs = dict.last_hidden_state
            hidden_states = dict.hidden_states
            return encoder_outputs, mask, hidden_states
        

        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states



class BartPromptSeq2SeqModel(Seq2SeqModel):

    def __init__(self, encoder: Seq2SeqEncoder, decoder: Seq2SeqDecoder, prompt_encoder, prompt_classifier):
        super().__init__(encoder, decoder)
        self.prompt_encoder = prompt_encoder
        self.prompt_classifier = prompt_classifier

    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, config, decoder_type=None, copy_gate=False,
                    use_encoder_mlp=False, use_recur_pos=False, tag_first=False):
        model = BartModel.from_pretrained(bart_model)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        encoder = model.encoder
        decoder = model.decoder

        if use_recur_pos:
            decoder.set_position_embedding(label_ids[0], tag_first)

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        prompt_encoder = PromptEncoder(model.config, deepcopy(encoder.embed_tokens))
        prompt_classifier = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, model.config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(model.config.hidden_size, model.config.vocab_size),
        )

        encoder = FBartEncoder(encoder)
        label_ids = sorted(label_ids)
        if decoder_type is None:
            assert copy_gate is False
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        elif decoder_type =='avg_score':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                              use_encoder_mlp=use_encoder_mlp)
        else:
            raise RuntimeError("Unsupported feature.")


        return cls(encoder=encoder, decoder=decoder, prompt_encoder=prompt_encoder, prompt_classifier=prompt_classifier)

    def prepare_state(self, src_tokens, src_seq_len=None, first=None, tgt_seq_len=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def prompt_mask_language_model(self, prompt_src, prompt_pos, prompt_mask_pos):
        prompt_pred = None
        embedding_outputs = self.encoder.bart_encoder.embed_tokens(prompt_src) * self.encoder.bart_encoder.embed_scale

        for batch_idx in range(prompt_pos.shape[0]):
            embedding_outputs[batch_idx][prompt_pos[batch_idx]] = self.prompt_encoder(prompt_src[batch_idx][prompt_pos[batch_idx]].unsqueeze(0)).squeeze(0)
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens=prompt_src, src_embs=embedding_outputs)
        
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, prompt_src, None, src_embed_outputs)
        
        encoder_pad_mask = state.encoder_mask          
        first = state.first
        for batch_idx in range(prompt_mask_pos.shape[0]):
            temp = self.prompt_classifier(state.encoder_output[batch_idx][prompt_mask_pos[batch_idx]])
            if prompt_pred is None:
                prompt_pred = temp.unsqueeze(0)
            else:
                prompt_pred = torch.cat((prompt_pred, temp.unsqueeze(0)), dim=0)
        return prompt_pred

    def forward(self, src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first, prompt_src=None, prompt_pos=None, prompt_mask_pos=None):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len, first, tgt_seq_len)
        decoder_output = self.decoder(tgt_tokens, state)

        returned_result = {}

        if prompt_pos is not None and prompt_src is not None and prompt_mask_pos is not None:
            returned_result['prompt_pred'] = self.prompt_mask_language_model(prompt_src, prompt_pos, prompt_mask_pos)
        

        if isinstance(decoder_output, torch.Tensor):
            returned_result['pred'] = decoder_output
        elif isinstance(decoder_output, (tuple, list)):
            returned_result['pred'] = decoder_output[0]
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")
        return returned_result