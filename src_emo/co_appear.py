from tqdm import tqdm
import json
from collections import defaultdict
import torch
from dataset_dbpedia import DBpedia


class TOCoAppear:
    """DSCooccur
    entity Symptom CO-Occurrence, disease symptom co-occurrence probability via training data.

    existing entity --> predicting entity
    """

    def __init__(self, kg):
        self.entity2id = kg.entity2id
        self.all_entity = list(self.entity2id.keys())
        self.eps = 1e-24
        self.entity_num = kg.num_entities
        self.id2entity = {
            id: entity
            for entity, id in self.entity2id.items()
        }
        self.total_cnt = 0
        self.cnt = 0
        trans_matrix = self._get_trans_matrix()
        self.trans_matrix = trans_matrix

    def _parse_trans(self, conv_entity_list):
        trans_pairs = list()

        for i in range(len(conv_entity_list)):
            main_role_name = conv_entity_list[i][0]
            main_role_emo = conv_entity_list[i][1]
            main_role_prob = conv_entity_list[i][2]
            next_index = len(conv_entity_list)
            trans_roles = conv_entity_list[i + 1: next_index]
            for trans_role in trans_roles:
                trans_role_name = trans_role[0]
                trans_role_emo = trans_role[1]
                trans_role_prob = trans_role[2]
                flag = False
                if len(main_role_emo)!= 0 and len(trans_role_emo)!= 0:
                    for emo in  main_role_emo[0:3] :
                        if emo in trans_role_emo[0:3]:
                            trans_pairs.append((main_role_name, trans_role_name))
                            self.cnt += 1
                            flag = True
                            break
                if flag == False:
                    self.total_cnt += 1

        return trans_pairs

    def _get_trans_matrix(self, freq_factor=0.2):
        entity_trans_pairs = list()

        entity2id = self.entity2id
        with open('data/redial/train_data_dbpedia_emo.jsonl', 'r', encoding='utf-8') as fin:
            for line in tqdm(fin):
                dialog = json.loads(line)
                conv_entity_list = []
                if len(dialog['messages']) == 0:
                    continue
                turns = []
                entity_list = []
                emo_list = []
                emo_probs_list = []
                messages = dialog['messages']
                turn_i = 0
                user_id, resp_id = dialog['initiatorWorkerId'], dialog['respondentWorkerId']

                while turn_i < len(messages):
                    worker_id = messages[turn_i]['senderWorkerId']
                    entity_turn = []
                    movie_turn = []
                    movie_name_turn = []
                    emo_turn = []
                    emo_prob_turn = []

                    turn_j = turn_i
                    while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == worker_id:
                        entity_ids = [entity2id[entity] for entity in messages[turn_j]['entity'] if entity in entity2id]
                        entity_turn.extend(entity_ids)

                        movie_ids = [entity2id[movie] for movie in messages[turn_j]['movie'] if movie in entity2id]
                        movie_turn.extend(movie_ids)
                        movie_name_turn.extend(messages[turn_j]['movie_name'])
                        turn_emotion = []
                        turn_emotion_probs = []
                        try:
                            turn_emotion_ = messages[turn_j]['emotion']
                            for emo, probs in turn_emotion_.items():
                                if probs > 0.1:
                                    turn_emotion.append(emo)
                                    turn_emotion_probs.append(probs)
                            if len(turn_emotion) == 0 and worker_id == user_id and len(entity_ids + movie_ids) != 0:
                                turn_emotion.append(list(turn_emotion_.keys())[0])
                                turn_emotion_probs.append(list(turn_emotion_.values())[0])
                        except:
                            pass
                        emo_turn.extend(turn_emotion)
                        emo_prob_turn.extend(turn_emotion_probs)

                        turn_j += 1
                    role = "seeker" if worker_id == user_id else "recommender"
                    turns.append({"role": role, "emo_turn": emo_turn,
                                  "emo_probs_turn": emo_prob_turn,
                                  "entity_turn": entity_turn, "movie_turn": movie_turn})
                    turn_i = turn_j

                for i in range(len(turns)):
                    turn = turns[i]
                    role = turn["role"]
                    next_turn = turns[i + 1] if i + 1 < len(turns) else None
                    emotion = turn["emo_turn"] if role == "seeker" else next_turn[
                        "emo_turn"] if next_turn is not None else []
                    emotion_porbs = turn["emo_probs_turn"] if role == "seeker" else next_turn[
                        "emo_probs_turn"] if next_turn is not None else []

                    if role == "seeker":
                        entity_list.extend(turn["movie_turn"] + turn["entity_turn"])
                        emo_list.extend([emotion] * (len(turn["movie_turn"]) + len(turn["entity_turn"])))
                        emo_probs_list.extend([emotion_porbs] * (len(turn["movie_turn"]) + len(turn["entity_turn"])))
                    else:
                        assert len(entity_list) == len(emo_list), "entity and emotion is not match"

                        entity_list.extend(turn["movie_turn"] + turn["entity_turn"])
                        emo_list.extend([emotion] * (len(turn["movie_turn"]) + len(turn["entity_turn"])))
                        emo_probs_list.extend([emotion_porbs] * (len(turn["movie_turn"]) + len(turn["entity_turn"])))


                conv_entity_list = list(zip(entity_list, emo_list, emo_probs_list))

                entity_trans_pairs += self._parse_trans(conv_entity_list)
        # -----------------------------------------------------------------------------------------
        # D, D
        trans_matrix = defaultdict(dict)
        for entity_pair in entity_trans_pairs:
            x, y = entity_pair[0], entity_pair[1]
            if y in trans_matrix[x]:
                trans_matrix[x][y] += 1
            else:
                trans_matrix[x][y] = 1
        for e in self.id2entity.keys():
            co_es = trans_matrix[e]
            co_num = sum(list(co_es.values()))
            if co_num == 0:
                print('here')
            nei_mvs_oh = torch.zeros(self.entity_num)
            for co_e, num in co_es.items():
                nei_mvs_oh[co_e] = num / co_num
            trans_matrix[e] = nei_mvs_oh

        return trans_matrix

    def get_top_k_predicted_entity(self,
                                   t_entity):
        return self.trans_matrix[t_entity]


if __name__ == "__main__":
    kg_class = DBpedia(dataset='redial_gen')
    toca = TOCoAppear(kg_class)
