
from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
import torch


class PromptAuxSeq2SeqLoss(LossBase):
    def __init__(self):
        super().__init__()

    def get_loss(self, tgt_tokens, tgt_seq_len, pred, prompt_pred=None, prompt_tgt=None):
        """

        :param tgt_tokens: bsz x max_len, [sos, tokens, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)

        loss_seq = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
        loss = loss_seq
        if prompt_pred is not None and prompt_tgt is not None:
            loss_prompt = F.cross_entropy(target=prompt_tgt, input=prompt_pred.transpose(1, 2))
            loss = loss_seq * 0.5 + loss_prompt * 0.5
        return loss

