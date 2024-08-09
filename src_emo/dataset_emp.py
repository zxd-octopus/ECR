import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import gpt2_special_tokens_dict
import random
from collections import defaultdict
import re

class CRSEmpDataset(Dataset):
    def __init__(
        self, dataset, split, tokenizer, debug=False,
        context_max_length=None, resp_max_length=None,
        infer = False, kg = None,sample = False, output = False, wk = False, wt = False, wn = False
    ):
        super(CRSEmpDataset, self).__init__()
        self.tokenizer = tokenizer
        self.debug = debug
        self.wk = wk
        self.wt = wt
        self.wn = wn

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length
        self.resp_max_length -= 1

        dataset_dir = os.path.join('data', dataset)
        if infer:
            sample_idx = None #os.path.join(dataset_dir, f'sample_data_processed_merge_LLM_scorer.jsonl')
            data_file = os.path.join(dataset_dir, f'{split}_data_processed_merge.jsonl')
        elif output:
            sample_idx = None # os.path.join(dataset_dir, f'sample_data_processed_merge_LLM_scorer.jsonl')
            data_file = os.path.join(dataset_dir, f'llama2_{split}_filtered.json')
        else:
            data_file = os.path.join(dataset_dir, f'movie_reviews_processed_{split}.json')# f'movie_reviews_processed_{split}_llama.json')

        with open(os.path.join(dataset_dir, 'common_template.json'), 'r', encoding='utf-8') as f:
            self.MASK_TEMPLATE = list(json.load(f).keys())

        self.data = []
        self.cnt = defaultdict(int)
        if output:
            self.get_data(data_file, dataset_dir, infer, kg, sample_idx, sample, output = output)
        else:
            self.prepare_data_infer(data_file, kg, sample_idx, sample, output = output) if infer else self.prepare_data(data_file, kg, output = output)



    def get_data(self, data_file, dataset_dir, infer, kg, sample_idx, sample, output):
        instruction = 'You are a recommender chatting with the user to provide movie recommendation. Please continue generating based on the First Sentence. Please utilize the information about Movie Name, Related Entities, and Related Knowledge from KG.'
        if not infer:
            self.prepare_data(data_file, kg, output=True)
            data_file = os.path.join(dataset_dir, 'llama2_valid_filtered.json')
            self.prepare_data(data_file, kg, output = True)
        else:
            self.prepare_data_infer(data_file, kg, sample_idx, sample, output=output)
        out_datas = []
        for sample in self.data:
            out_data = {}
            out_data['input'] = sample['context'].replace('<|endoftext|>','\n')
            out_data['output'] = " " if infer else sample['resp']
            out_data['instruction'] = instruction
            out_datas.append(out_data)
        if infer:
            file_name = 'data/redial/llama_test.json'
        else:
            file_name = 'data/redial/llama_train.json'
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(out_datas, f, ensure_ascii=False)



    def entity2name(self, entity):
        # if '(' in entity:
        #     entity = entity.split('(')[0].strip()
        if '/' in entity:
            entity = entity.split('/')[-1][:-1].strip()
        if '_' in entity:
            entity = entity.replace('_', ' ').strip()
        return entity

    def prepare_data(self, data_file, kg, output = False):
        with open("data/redial/mv2e_review.json", 'r', encoding='utf-8') as f:
            mv2e_review = json.load(f)
        with open('data/redial/review_e_filted.json', encoding='utf-8') as f:
            eset = json.load(f)
        with open(data_file, 'r', encoding='utf-8') as f:
            reviews = json.load(f)

            cnt = 0
            for review in tqdm(reviews):
                cnt+=1
                if cnt== 19500:
                    print("here")
                movie_name = review["movie_name"]

                e_review = review["e_review"]
                e_review = [kg.entity2name(e, True) for e in e_review]
                e_review_noise = mv2e_review[movie_name.split('(')[0].strip()]
                e_review_noise = [kg.entity2name(e[0], True) for e in e_review_noise]
                for m in e_review:
                    if m in e_review_noise:
                        e_review_noise.remove(m)
                if len(e_review_noise) > 2:
                    e_review_noise = random.sample(e_review_noise, 2)
                # e_review = e_review[:2]
                # e_review = e_review + e_review_noise
                # random.shuffle(e_review)

                e_relation_ = review['e_relation']
                # e_relation_org = []
                # for e,r in e_relation_:
                #     if r in ['starring', 'writer', 'productionCompany', 'director', 'musicComposer']:#"self loop",
                #         e_relation_org.append([e,r])
                e_relation_org = e_relation_
                e_relation_noise = kg.get_one_hop_neighborhood(movie_name, False)
                e_relation_noise = [[e, r] for e,r in e_relation_noise.items()]
                for m in e_relation_org:
                    if m in e_relation_noise:
                        e_relation_noise.remove(m)
                if len(e_relation_noise) > 1:
                    e_relation_noise = random.sample(e_relation_noise, 1)
                # if len(e_relation_org) > 1:
                #     e_relation_org = random.sample(e_relation_org, 1)
                # e_relation = e_relation_org + e_relation_noise
                e_relation = e_relation_org
                # random.shuffle(e_relation)
                # neighborhood_list = kg.get_one_hop_neighborhood(movie_name)
                # plot_str = kg.get_movie_plot(movie_name)

                if len(e_review) == 0:
                    e_review = ["None"]

                context = 'Movie Name: ' + movie_name
                context += self.tokenizer.eos_token
                if self.wt:
                    e_review.extend([e for e,r in e_relation])
                if not self.wk:
                    # related_entities = random.sample(related_entities, 5)
                    context += 'Related Entities: ' + ", ".join(e_review)
                    context += self.tokenizer.eos_token
                    if not self.wt:
                        e_kg_template = '{}\'s {} is {}'
                        e_kg = []
                        for e,r in e_relation:
                            e_kg.append(e_kg_template.format(movie_name, r, e))
                        if len(e_kg) == 0:
                            e_kg = ["None"]
                        context += 'Related Knowledge from KG: ' + ", ".join(e_kg)
                        context += self.tokenizer.eos_token
                context += 'First Sentence: ' + random.choice(self.MASK_TEMPLATE).split(':')[1].strip().replace('<movie>', movie_name)



                # context += 'Plot: ' + plot_str
                # context += self.tokenizer.eos_token


                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[:(self.context_max_length-1)]
                context_ids += [self.tokenizer.eos_token_id]

                # resp = review["title"] + ' ' + review["content"]
                resp = review["content"]

                if not self.wt and not self.wk:

                    if len(e_relation_org)> 0:
                        resp = re.split("([.?!])", resp)
                        pos_org, r_org = -3, -3
                        for e,r in e_relation_org:
                            rand_p = random.randint(1, 10)
                            if rand_p > 9:
                                copy_pos = 0
                                for copy_pos in range(len(resp)):
                                    if e.lower() in resp[copy_pos].lower():
                                        break
                                if copy_pos == len(resp):
                                    copy_pos = random.randint(0, copy_pos)
                                if copy_pos != 0:
                                    movie_name = random.choice(["This movie", "It", "This film", movie_name.split('(')[0].strip() ,"The movie", "The film"])
                                if (pos_org == copy_pos or pos_org + 1 == copy_pos) and r_org == r:
                                    resp[pos_org] = resp[pos_org][:-1]+ " and {}".format(e)+ resp[pos_org][-1]
                                    copy_pos = pos_org
                                else:
                                    resp.insert(copy_pos, e_kg_template.format(movie_name.split('(')[0].strip(), r, e)+'.')
                                pos_org, r_org = copy_pos, r
                        resp = ''.join(resp)

                with self.tokenizer.as_target_tokenizer():
                    resp_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(resp))
                    resp_ids = resp_ids[:self.resp_max_length]
                    resp_ids.append(self.tokenizer.eos_token_id)
                data = {
                    'context': context if output else context_ids,
                    'resp': resp  if output else resp_ids
                }
                self.data.append(data)

    def prepare_data_infer(self, data_file, kg, sample_idx, sample, output = False):
        with open("data/redial/mv2e_review.json", 'r', encoding='utf-8') as f:
            mv2e_review = json.load(f)

        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:512]
            if sample:
                with open(sample_idx, 'r', encoding='utf-8') as f_idx:
                    sample_idxs = json.load(f_idx)
                    sample_idxs = list(sample_idxs.keys())
                lines = [lines[int(i)] for i in sample_idxs[:1000]]

            for line in tqdm(lines):
                dialog = json.loads(line)
                if "resp_all" not in dialog.keys():
                    continue

                if len(dialog["rec_gen"]) == 0:
                    continue

                movie_name = [kg.entity2name(mv) for mv in set(dialog["rec_gen"])]

                neighborhood_kg = []
                # plot_str = ""
                for mv in set(dialog["rec_gen"]):
                    neighborhood = kg.get_one_hop_neighborhood(mv)
                    e_kg_template = '{}\'s {} is {}'
                    for e, r in neighborhood.items():
                        neighborhood_kg.append(e_kg_template.format(kg.entity2name(mv), r, e))
                neighborhood_kg = list(set(neighborhood_kg))
                    # plot_str += kg.get_movie_plot(mv)
                e_review = []
                # for mv not in training set
                for mv in movie_name:
                    try:
                        e_review.extend(mv2e_review[mv.split('(')[0].strip()])
                    except:
                        pass
                e_review = [self.entity2name(e[0]) for e in e_review]
                # flag = True
                # for mv in movie_name:
                #     if mv.split('(')[0].strip() in mv2e_review.keys():
                #         flag = False
                # if flag == False or len(movie_name) == 0:
                #     continue

                if self.wn:
                    e_review_len = 8
                    kg_len = 4
                else:
                    e_review_len = 4
                    kg_len = 2
                # kg_e_len = 2 if len(e_review) >= 4 else 6-len(e_review)
                if len(neighborhood_kg) > kg_len:
                    neighborhood_kg = random.sample(neighborhood_kg, kg_len)
                #self.cnt[len(neighborhood_list)] += 1
                # review_e_len = 4 if len(neighborhood_list) >= 2 else 6 - len(neighborhood_list)
                if len(e_review) > e_review_len:
                    e_review = random.sample(e_review, e_review_len)
                if len(e_review) == 0:
                    e_review = ["None"]
                if len(neighborhood_kg) == 0:
                    neighborhood_kg = ["None"]
                if self.wt:
                    for kg_str in neighborhood_kg:
                        kg_e = kg_str.split(" is ")[-1]
                        e_review.append(kg_e)
                context = 'Movie Name: ' + ", ".join(movie_name)
                context += self.tokenizer.eos_token
                if not self.wk:
                    context += 'Related Entities: ' + ", ".join(e_review)
                    context += self.tokenizer.eos_token
                    if not self.wt:
                        context += 'Related Knowledge from KG: ' + ", ".join(neighborhood_kg)
                        context += self.tokenizer.eos_token
                context += 'First Sentence: ' + dialog['resp_all'].replace('<movie>', ", ".join(movie_name))


                # context += 'Plot: ' + plot_str
                # context += self.tokenizer.eos_token

                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[:(self.context_max_length-1)]
                context_ids += [self.tokenizer.eos_token_id]

                resp = dialog["resp_gth"]
                with self.tokenizer.as_target_tokenizer():
                    resp_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(resp))
                    resp_ids = resp_ids[:self.resp_max_length]
                    resp_ids.append(self.tokenizer.eos_token_id)
                data = {
                    'context': context if output else context_ids,
                    'resp':resp  if output else resp_ids
                }
                self.data.append(data)

        self.cnt = sorted(self.cnt.items(), key=lambda d: d[0])
        print("here")


    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class CRSEmpDataCollator:
    def __init__(
        self, tokenizer, device, gen=False, use_amp=False, debug=False, ignore_pad_token_for_loss=True,
        context_max_length=None, resp_max_length=None,
        prompt_tokenizer=None
    ):
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.device = device
        self.use_amp = use_amp
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.gen = gen
        self.debug = debug

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        resp_batch = []
        context_len_batch = []

        if self.gen:
            self.tokenizer.padding_side = 'left'
            for data in data_batch:
                context_ids = data['context']
                context_ids = context_ids[-(self.context_max_length):]
                context_len_batch.append(len(context_ids))
                context_batch['input_ids'].append(context_ids)
                resp_batch.append(data['resp'])
        else:
            self.tokenizer.padding_side = 'right'

            for data in data_batch:
                input_ids = data['context'] + data['resp']
                input_ids = input_ids[-self.context_max_length:]
                context_batch['input_ids'].append(input_ids)

        input_batch = {}

        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.context_max_length
        )
        if not self.gen:
            resp_batch = context_batch['input_ids']
            resp_batch = [[token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in resp] for resp
                          in resp_batch]
            input_batch['resp'] = torch.as_tensor(resp_batch, device=self.device)
        else:
            input_batch['resp'] = resp_batch
            input_batch['context_len'] = context_len_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch
        return input_batch


if __name__ == '__main__':
    from dataset_dbpedia import DBpedia
    from pprint import pprint

    debug = False
    gen = True
    device = torch.device('cpu')
    dataset = 'redial'

    kg = DBpedia(dataset=dataset, debug=debug)

    model_name_or_path = "save/dialogpt/"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)

    # prompt_tokenizer = AutoTokenizer.from_pretrained('../utils/tokenizer/roberta-base')

    # train_dataset = CRSEmpDataset(
    #     dataset, 'train', tokenizer,
    #     context_max_length=150, resp_max_length=150, kg = kg, output = True
    # )

    infer_dataset = CRSEmpDataset(
        'redial_gen', 'test', tokenizer,
        context_max_length=150, resp_max_length=150, kg = kg, infer = True, output = True , sample = True
    )