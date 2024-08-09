import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from config import  Emo_loss_weight_dict

from utils import padded_tensor
loss_weight_dic = {1:1.5, 0: 1.0, -1:0.3}
mse_dic = {1:1, -1:0, 0: -1}

class CRSRecDataset(Dataset):
    def __init__(
        self, dataset, split, tokenizer, debug=False,
        context_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None,
        emotion_max_length = None, use_resp=False,
        emo2idx = None,remove_neg = False,kg = None,
            toca = None
    ):
        super(CRSRecDataset, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.use_resp = use_resp
        self.emo2idx = emo2idx
        self.kg = kg
        self.toca = toca

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.emotion_max_length = emotion_max_length
        if self.emotion_max_length is None:
            self.emotion_max_length = 2

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length
        self.prompt_max_length -= 1

        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        dataset_dir = os.path.join('data', dataset)
        data_file = os.path.join(dataset_dir, f'{split}_data_processed.jsonl')
        self.data = []
        self.entity_max_nei = 3
        self.prepare_data(data_file, remove_neg)

    def prepare_data(self, data_file, remove_neg):

        if remove_neg and ("test" in data_file or "valid" in data_file):
            sample_threshold = -1
        else:
            sample_threshold = -2
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]
            print(json.loads(lines[1])["resp"])

            for line in tqdm(lines):
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:
                    continue
                if len(dialog['context']) == 1 and dialog['context'][0] == '':
                    continue

                context = ''
                prompt_context = ''

                for i, utt in enumerate(dialog['context']):
                    if utt == '':
                        continue
                    if i % 2 == 0:
                        context += 'User: '
                        prompt_context += 'User: '
                    else:
                        context += 'System: '
                        prompt_context += 'System: '
                    context += utt
                    context += self.tokenizer.eos_token
                    prompt_context += utt
                    prompt_context += self.prompt_tokenizer.sep_token

                if context == '':
                    continue
                if self.use_resp:
                    if i % 2 == 0:
                        resp = 'System: '
                    else:
                        resp = 'User: '
                    resp += dialog['resp']
                    context += resp + self.tokenizer.eos_token
                    prompt_context += resp + self.prompt_tokenizer.sep_token

                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[-self.context_max_length:]

                prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context))
                prompt_ids = prompt_ids[-self.prompt_max_length:]
                prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)

                emos_entities = []
                emos_entities_probs = []
                for emos_entity, emotion_probs_entity in zip(dialog['emotion_entity'][-self.entity_max_length:],
                                                             dialog['emotion_probs_entity'][-self.entity_max_length:]):
                    emos_entities.append([self.emo2idx[emo] for emo in emos_entity][-self.emotion_max_length:])
                    emos_entities_probs.append(emotion_probs_entity[-self.emotion_max_length:])

                nei_mvs = [] # (entity_len，n_entity)
                for e in dialog['entity'][-self.entity_max_length:]:
                    # ei_mvs_idx = torch.tensor(self.kg.get_nei_mv(e)) # (batch_size, entity_len)
                    nei_mvs_idx = self.toca.get_top_k_predicted_entity(e) # (batch_size, entity_len)
                    # try:
                    #     nei_mvs_oh = F.one_hot(nei_mvs_idx, num_classes=self.kg.num_entities)
                    #     nei_mvs_oh = nei_mvs_oh.sum(-2) #[n_entity]
                    # except:
                    #     nei_mvs_oh = torch.zeros(self.kg.num_entities)
                    nei_mvs.append(nei_mvs_idx)


                rec_weight = 0
                for i in range(len(dialog['rec_weight'])):
                    rec_weight += dialog['rec_weight_w'][i]*(Emo_loss_weight_dict[dialog['rec_weight'][i]])

                for item, sentiment  in zip(dialog['rec'], dialog['sentiment_loss']):
                    if sentiment > sample_threshold:
                        # if "test" in data_file or "valid" in data_file:
                        #     if item in dialog['entity'][-self.entity_max_length:]:
                        #         continue
                        data = {
                            'context': context_ids,
                            'entity': dialog['entity'][-self.entity_max_length:],
                            'emos_entities': emos_entities,
                            'emos_entities_probs':emos_entities_probs,
                            'rec': item,
                            'rec_weight': rec_weight,
                            'sentiment_loss': sentiment,
                            'prompt': prompt_ids,
                            'nei_mvs' : nei_mvs,
                            # 'nei_mvs_emos': nei_mvs_emos,
                            # 'nei_mvs_probs': nei_mvs_probs,
                        }
                        self.data.append(data)

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


class CRSRecDataCollator:
    def __init__(
        self, tokenizer, device, pad_entity_id, use_amp=False, debug=False,
        context_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None,
        pad_emotion_id = None, emotion_max_length = None,
        entity_num = None, like_score = 2.0, dislike_score = 0.3, notsay_score = 1.0, nei_mv_max_length = 20,n_entity = None,
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.entity_num = entity_num
        self.loss_weight_dic = {1: 1, -1: 0, 0: -1}
        self.loss_weight_dic[1] = like_score
        self.loss_weight_dic[-1] = dislike_score
        self.loss_weight_dic[0] = notsay_score

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length

        self.pad_entity_id = pad_entity_id
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        self.pad_emotion_id = pad_emotion_id
        self.emotion_max_length = emotion_max_length
        self.nei_mv_max_length = nei_mv_max_length
        if self.emotion_max_length is None:
            self.emotion_max_length = self.tokenizer.model_max_length
        self.n_entity = n_entity

        # self.rec_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('Recommend:'))

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        label_batch = []
        label_sentiment_batch = []
        label_weight_batch = []
        emotion_batch = []
        emotion_probs_batch = []
        RMSE_met = []
        context_entities_mask_batch = []
        nei_mv_batch = []


        for data in data_batch:
            # input_ids = data['context'][-(self.context_max_length - len(self.rec_prompt_ids)):] + self.rec_prompt_ids
            input_ids = data['context']
            context_batch['input_ids'].append(input_ids)

            entity_batch.append(data['entity'])
            context_entities_mask = torch.zeros(self.entity_num, device = self.device)
            ce_index = torch.tensor(data['entity'] ,device = self.device, dtype=torch.long)
            context_entities_mask  = context_entities_mask.scatter(0, ce_index, 1)
            context_entities_mask_batch.append(context_entities_mask)

            label_batch.append(data['rec'])
            prompt_batch['input_ids'].append(data['prompt'])
            emotion_batch.append(data['emos_entities'])
            emotion_probs_batch.append(data['emos_entities_probs'])
            nei_mv_batch.append(data['nei_mvs'])
            label_sentiment_batch.append(self.loss_weight_dic[data['sentiment_loss']])
            RMSE_met.append(mse_dic[data['sentiment_loss']])
            label_weight_batch.append(data['rec_weight'])

        input_batch = {}
        context_entities_mask_batch = torch.stack(context_entities_mask_batch)
        input_batch['context_entities_mask_batch'] = context_entities_mask_batch.bool()

        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.context_max_length
        )
        context_batch['rec_labels'] = label_batch
        context_batch['rec_sentiment_loss'] = torch.tensor(label_sentiment_batch,dtype=torch.float,device = self.device)
        context_batch['label_weight_batch'] = torch.tensor(label_weight_batch,dtype=torch.float,device = self.device)
        input_batch['RMSE_met'] = torch.tensor(RMSE_met, dtype=torch.float, device=self.device)
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch

        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, padding=self.padding, max_length=self.prompt_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch

        entity_batch = padded_tensor(entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device)
        max_len = entity_batch.size(-1)
        nei_mvs = []
        for nei_mv in nei_mv_batch:
            if len(nei_mv) != 0:
                pad_len = max_len - len(nei_mv)
                pad_nei_mv = torch.zeros(pad_len, self.n_entity, device=self.device,
                                               dtype=torch.float)
                nei_mv = torch.stack(nei_mv, dim=0)
                nei_mv_ = torch.cat((nei_mv.to(self.device), pad_nei_mv), 0)
            else:
                nei_mv_ = torch.zeros(max_len, self.n_entity, device=self.device,
                                               dtype=torch.float)
            nei_mvs.append(nei_mv_)
        nei_mvs = torch.stack(nei_mvs, dim=0) # (batch_size, entity_len，n_entity)

        emotions_batch = []
        emotions_probs_batch = []
        for emotion,emotion_probs in zip(emotion_batch,emotion_probs_batch):
            pad_len = max_len - len(emotion)
            try:
                emotion = padded_tensor(emotion, pad_idx=self.pad_emotion_id, pad_tail=True, device=self.device,
                                        fixed_len=self.emotion_max_length)
                emotion_probs = padded_tensor(emotion_probs, pad_idx=0, pad_tail=True, device=self.device,
                                              fixed_len=self.emotion_max_length,dtype_ = torch.float)

            except:
                try:
                    emotion = torch.tensor(emotion, device=self.device, dtype=torch.long)
                    emotion_probs = torch.tensor(emotion_probs, device=self.device, dtype=torch.float)

                except:
                    print(emotion)
            pad_tensor = torch.ones(pad_len, self.emotion_max_length, device=self.device,
                                    dtype=torch.long) * self.pad_emotion_id
            pad_tensor_probs = torch.zeros(pad_len, self.emotion_max_length, device=self.device,
                                           dtype=torch.float)

            emotion_batch_ = torch.cat((emotion, pad_tensor), 0)
            emotions_batch.append(emotion_batch_)
            emotion_probs_batch_ = torch.cat((emotion_probs, pad_tensor_probs), 0)
            emotions_probs_batch.append(emotion_probs_batch_)

        emotions_batch = torch.stack(emotions_batch, dim=0)
        emotions_probs_batch = torch.stack(emotions_probs_batch, dim=0)


        input_batch['entity'] = entity_batch
        input_batch['emotion'] = emotions_batch
        input_batch['emotion_probs'] = emotions_probs_batch
        input_batch['nei_mvs'] = nei_mvs


        return input_batch


if __name__ == '__main__':
    from dataset_dbpedia import DBpedia
    from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
    from pprint import pprint

    debug = True
    device = torch.device('cpu')
    dataset = 'inspired'

    kg = DBpedia(dataset, debug=debug).get_entity_kg_info()

    model_name_or_path = "../utils/tokenizer/dialogpt-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    prompt_tokenizer = AutoTokenizer.from_pretrained('../utils/tokenizer/roberta-base')
    prompt_tokenizer.add_special_tokens(prompt_special_tokens_dict)

    dataset = CRSRecDataset(
        dataset=dataset, split='test', tokenizer=tokenizer, debug=debug,
        prompt_tokenizer=prompt_tokenizer
    )
    for i in range(len(dataset)):
        if i == 3:
            break
        data = dataset[i]
        print(data)
        print(tokenizer.decode(data['context']))
        print(prompt_tokenizer.decode(data['prompt']))
        print()

    data_collator = CRSRecDataCollator(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=prompt_tokenizer
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=data_collator,
    )

    input_max_len = 0
    entity_max_len = 0
    for batch in tqdm(dataloader):
        if debug:
            pprint(batch)
            exit()

        input_max_len = max(input_max_len, batch['context']['input_ids'].shape[1])
        entity_max_len = max(entity_max_len, batch['entity'].shape[1])

    print(input_max_len)
    print(entity_max_len)
    # (767, 26), (645, 29), (528, 16) -> (767, 29)
    # inspired: (993, 25), (749, 20), (749, 31) -> (993, 31)
