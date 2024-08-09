import math

import torch
import json

class RecEvaluator:
    def __init__(self, k_list=None, device=torch.device('cpu')):
        if k_list is None:
            k_list = [1, 10, 50]
        self.k_list = k_list
        self.device = device

        self.metric = {}
        self.metric_add = {}
        self.reset_metric()

        self.log_file = open("save/redial_rec/rec.json", 'w', buffering=1)
        self.log_cnt = 0

        with open('data/redial/entity2id.json', 'r', encoding='utf-8') as f:
            entity2id = json.load(f)
        self.id2entity = {idx:entity for entity,idx in entity2id.items()}

    def get_mv_name(self,ename):
        mname = ename.split('/')[-1][:-1]
        mname = mname.replace('_',' ')
        return mname

    def evaluate(self, logits, labels, log = False):
        for logit, label in zip(logits, labels):
            for k in self.k_list:
                self.metric[f'recall@{k}'] += self.compute_recall(logit, label, k)
                self.metric[f'mrr@{k}'] += self.compute_mrr(logit, label, k)
                self.metric[f'ndcg@{k}'] += self.compute_ndcg(logit, label, k)
            self.metric['count'] += 1
        if log:
            for pred, label in zip(logits, labels.tolist()):
                max_mv = pred[0]
                pre_name = self.id2entity[max_mv]
                gth_name = self.id2entity[label]
                self.log_file.write(json.dumps({
                    'pred': pre_name,
                    'label': gth_name
                }, ensure_ascii=False) + '\n')

    def evaluate_mse(self, label_logits, labels, mse_met):
        for logit, label, gth in zip(label_logits.tolist(), labels.tolist(), mse_met[0]):
            if gth != -1:
                self.metric_add['mse'] += torch.pow(logit[label] - gth, 2)
                self.metric_add['count_mse'] += 1
            if gth == 0:
                self.metric_add['mse_neg'] += torch.pow(logit[label] - gth, 2)
                self.metric_add['count_mse_neg'] += 1

    def evaluate_AUC(self, logits, labels, mse_met):
        pre = []
        for logit, label in zip(logits.tolist(), labels.tolist()):
            pre.append(logit[label])
        pos = [i for i in range(len(mse_met[0])) if mse_met[0][i] == 1]
        neg = [i for i in range(len(mse_met[0])) if mse_met[0][i] == 0]
        auc = 0
        for i in pos:
            for j in neg:
                if pre[i] > pre[j]:
                    auc += 1
                elif pre[i] == pre[j]:
                    auc += 0.5
        try:
            self.metric_add['auc_batch'] +=  auc / (len(pos) * len(neg))
            self.metric_add['batch_count'] += 1
        except:
            pass



    def evaluate_true_recall(self, logits, labels, mse_met):
        for logit, label, gth in zip(logits, labels, mse_met[0]):
            if gth == 1:
                for k in self.k_list:
                    self.metric_add[f'recall_true@{k}'] += self.compute_recall(logit, label, k)
                self.metric_add['count_true'] += 1

    def compute_recall(self, rank, label, k):
        return int(label in rank[:k])

    def compute_mrr(self, rank, label, k):
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / (label_rank + 1)
        return 0

    def compute_ndcg(self, rank, label, k):
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / math.log2(label_rank + 2)
        return 0

    def reset_metric(self):
        for metric in ['recall', 'ndcg', 'mrr']:
            for k in self.k_list:
                self.metric[f'{metric}@{k}'] = 0
        self.metric['count'] = 0

        for k in self.k_list:
            self.metric_add[f'recall_true@{k}'] = 0
        self.metric_add['count_mse'] = 0
        self.metric_add['mse'] = 0
        self.metric_add['count_true'] = 0
        self.metric_add['mse_neg'] = 0
        self.metric_add['count_mse_neg'] = 0
        self.metric_add['auc_batch'] = 0
        self.metric_add['batch_count'] = 0



    def report(self):
        report = {}
        report_add = {}
        for k, v in self.metric.items():
            report[k] = torch.tensor(v, device=self.device)[None]
        for k, v in self.metric_add.items():
            report_add[k] = torch.tensor(v, device=self.device)[None]
        return report, report_add
