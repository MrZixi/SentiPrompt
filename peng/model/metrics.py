from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
from collections import Counter
import os

class Seq2SeqSpanMetric(MetricBase):
    def __init__(self, eos_token_id, num_labels, opinion_first=True, output_bad_cases=None):
        super(Seq2SeqSpanMetric, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.word_start_index = num_labels + 2  # +2, shift for sos and eos
        self.output_case_log = os.path.join(output_bad_cases, 'pred.log') if output_bad_cases is not None else None
        self.output_tgt_log = os.path.join(output_bad_cases, 'tgt.log') if output_bad_cases is not None else None
        if self.output_case_log is not None:
            if os.path.exists(self.output_case_log):
                os.remove(self.output_case_log)
            if os.path.exists(self.output_tgt_log):
                os.remove(self.output_tgt_log)

        self.ae_oe_fp = 0
        self.ae_oe_tp = 0
        self.ae_oe_fn = 0
        self.triple_fp = 0
        self.triple_tp = 0
        self.triple_fn = 0
        self.em = 0
        self.invalid_token = 0
        self.invalid_size = 0
        self.invalid_order = 0
        self.total = 0
        self.ae_sc_fp = 0
        self.ae_sc_tp = 0
        self.ae_sc_fn = 0
        assert opinion_first is False, "Current metric only supports aspect first"

        self.opinin_first = opinion_first

    def evaluate(self, target_span, pred, tgt_tokens):
        self.total += pred.size(0)
        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # delete </s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1)  # bsz
        target_seq_len = (target_seq_len - 2).tolist()
        pred_spans = []
        if self.output_case_log is not None:
            with open(self.output_case_log, 'a') as f:
                temp_preds = pred.cpu().numpy().tolist()
                if not len(temp_preds) == 1:
                    for temp_pred in temp_preds:
                        temp_pred = [x for x in temp_pred if x != 1]
                        f.write(str(temp_pred))
                        f.write('\n')
            with open(self.output_tgt_log, 'a') as f:
                temp_tgts = target_span.tolist()
                if not len(temp_tgts) == 1:
                    for temp_tgt in temp_tgts:
                        temp_tgt = [x for x in temp_tgt if x != 1]
                        f.write(str(temp_tgt))
                        f.write('\n')
        # print(tgt_tokens)
        # print(pred)
        # print(target_span)
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            em = 0
            ps = ps[:pred_seq_len[i]]
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(
                    tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item() == target_seq_len[i])
            self.em += em
            invalid_size = 0
            invalid_order = 0
            invalid_token = 0
            pairs = []
            cur_pair = []
            token_len = len(tgt_tokens[i])
            if len(ps):
                for index, j in enumerate(ps):
                    if j < self.word_start_index:
                        cur_pair.append(j)
                        token_invalid = lambda k: cur_pair[k] > token_len + 4 or cur_pair[k] < 5
                        if len(cur_pair) != 5 or cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3] or token_invalid(0) or token_invalid(1) or token_invalid(2) or token_invalid(3) or cur_pair[4] < 2 or cur_pair[4] > 4:
                            if len(cur_pair) != 5:
                                invalid_size = 1
                            elif cur_pair[0] > cur_pair[1] or cur_pair[2] > cur_pair[3]:
                                invalid_order = 1
                            elif token_invalid(0) or token_invalid(1) or token_invalid(2) or token_invalid(3) or cur_pair[4] < token_len or cur_pair[4] > token_len + 3:
                                # print(ts)
                                # print(token_len)
                                # print(cur_pair)
                                invalid_token = 1
                        else:
                            pairs.append(tuple(cur_pair))
                        cur_pair = []
                    else:
                        cur_pair.append(j)
            pred_spans.append(pairs.copy())
            self.invalid_size += invalid_size
            self.invalid_token += invalid_token
            self.invalid_order += invalid_order

            oe_ae_target = [tuple(t[:4]) for t in ts]
            oe_ae_pred = [p[:4] for p in pairs]

            oe_ae_target_counter = Counter(oe_ae_target)
            oe_ae_pred_counter = Counter(oe_ae_pred)
            tp, fn, fp = _compute_tp_fn_fp(set(list(oe_ae_pred_counter.keys())),
                                           set(list(oe_ae_target_counter.keys())))
            self.ae_oe_fn += fn
            self.ae_oe_fp += fp
            self.ae_oe_tp += tp

            # note aesc
            ae_sc_target = [(t[0], t[1], t[-1]) for t in ts]
            ae_sc_pred = [(p[0], p[1], p[-1]) for p in pairs]
            asts = set([tuple(t) for t in ae_sc_target])
            asps = set(ae_sc_pred)
            for p in list(asps):  # pairs is a 5-tuple
                if p in asts:
                    asts.remove(p)
                    self.ae_sc_tp += 1
                else:
                    self.ae_sc_fp += 1
            self.ae_sc_fn += len(asts)

            ts = set([tuple(t) for t in ts])
            ps = set(pairs)
            for p in list(ps):
                if p in ts:
                    ts.remove(p)
                    self.triple_tp += 1
                else:
                    self.triple_fp += 1

            self.triple_fn += len(ts)

    def get_metric(self, reset=True):
        res = {}
        f, pre, rec = _compute_f_pre_rec(1, self.triple_tp, self.triple_fn, self.triple_fp)

        res['triple_f'] = round(f, 4)*100
        res['triple_rec'] = round(rec, 4)*100
        res['triple_pre'] = round(pre, 4)*100

        f, pre, rec = _compute_f_pre_rec(1, self.ae_oe_tp, self.ae_oe_fn, self.ae_oe_fp)

        res['oe_ae_f'] = round(f, 4)*100
        res['oe_ae_rec'] = round(rec, 4)*100
        res['oe_ae_pre'] = round(pre, 4)*100

        f, pre, rec = _compute_f_pre_rec(1, self.ae_sc_tp, self.ae_sc_fn, self.ae_sc_fp)
        res["ae_sc_f"] = round(f, 4)*100
        res["ae_sc_rec"] = round(rec, 4)*100
        res["ae_sc_pre"] = round(pre, 4)*100

        res['em'] = round(self.em / self.total, 4)
        res['invalid_token'] = round(self.invalid_token / self.total, 4)
        res['invalid_size'] = round(self.invalid_size / self.total, 4)
        res['invalid_order'] = round(self.invalid_order / self.total, 4)
        if reset:
            self.ae_oe_fp = 0
            self.ae_oe_tp = 0
            self.ae_oe_fn = 0
            self.triple_fp = 0
            self.triple_tp = 0
            self.triple_fn = 0
            self.em = 0
            self.invalid_token = 0
            self.invalid_size = 0
            self.invalid_order = 0
            self.total = 0
            self.ae_sc_fp = 0
            self.ae_sc_tp = 0
            self.ae_sc_fn = 0

        return res


def _compute_tp_fn_fp(ps, ts):
    ps = ps.copy()
    tp = 0
    fp = 0
    fn = 0
    if isinstance(ts, set):
        ts = {key: 1 for key in list(ts)}
    if isinstance(ps, set):
        ps = {key: 1 for key in list(ps)}
    for key in ts.keys():
        t_num = ts[key]
        if key not in ps:
            p_num = 0
        else:
            p_num = ps[key]
        tp += min(p_num, t_num)
        fp += max(p_num - t_num, 0)
        fn += max(t_num - p_num, 0)
        if key in ps:
            ps.pop(key)
    fp += sum(ps.values())
    return tp, fn, fp
