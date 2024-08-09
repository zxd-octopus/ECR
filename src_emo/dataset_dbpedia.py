import json
import os

import torch
from loguru import logger
from torch_geometric.typing import SparseTensor
from collections import defaultdict
from tqdm import tqdm

class DBpedia:
    def __init__(self, dataset, debug=False):
        self.debug = debug
        if dataset is None:
            self.dataset_dir = ""
        else:
            self.dataset_dir = os.path.join('data', dataset)
        with open(os.path.join(self.dataset_dir, 'dbpedia_subkg.json'), 'r', encoding='utf-8') as f:
            self.entity_kg = json.load(f)
        with open(os.path.join(self.dataset_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.entity2id = json.load(f)

        self.entity2id_ = {}
        for key, value in self.entity2id.items():
            key = self.entity2name(key,True)
            self.entity2id_[key] = value
        self.id2entity = {idx: entity for entity, idx in self.entity2id_.items()}


        with open(os.path.join(self.dataset_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            self.relation2id = json.load(f)

        self.id2relation = {idx: relation for relation, idx in self.relation2id.items()}

        with open(os.path.join(self.dataset_dir, 'item_ids.json'), 'r', encoding='utf-8') as f:
            self.item_ids = json.load(f)

        self._process_entity_kg()

    def _process_entity_kg(self):
        self.neighborhood = defaultdict(list)
        edge_list = set()  # [(entity, entity, relation)]
        for entity in self.entity2id.values():
            if str(entity) not in self.entity_kg:
                continue
            for relation_and_tail in self.entity_kg[str(entity)]:
                self.neighborhood[entity].append(relation_and_tail[1])
                self.neighborhood[relation_and_tail[1]].append(entity)

                edge_list.add((entity, relation_and_tail[1], relation_and_tail[0]))
                edge_list.add((relation_and_tail[1], entity, relation_and_tail[0]))
        edge_list.add((max(self.entity2id.values()) + 1, max(self.entity2id.values()) + 1, 0))
        edge_list = list(edge_list)

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        self.edge_index = edge[:, :2].t().cuda()
        self.edge_type = edge[:, 2].cuda()
        self.edge_index = SparseTensor(row=self.edge_index[0], col=self.edge_index[1], value=self.edge_type)
        self.num_relations = len(self.relation2id)
        self.pad_entity_id = max(self.entity2id.values()) + 1
        self.num_entities = max(self.entity2id.values()) + 2

        if self.debug:
            logger.debug(
                f'#edge: {len(edge)}, #relation: {self.num_relations}, '
                f'#entity: {self.num_entities}, #item: {len(self.item_ids)}'
            )

    def get_entity_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'pad_entity_id': self.pad_entity_id,
            'item_ids': self.item_ids,
        }
        return kg_info

    def entity2name(self, entity, flag = False):
        if flag:
            if '(' in entity:
                entity = entity.split('(')[0].strip()
        if '/' in entity:
            entity = entity.split('/')[-1][:-1].strip()
        if '_' in entity:
            entity = entity.replace('_', ' ').strip()
        return entity

    def get_one_hop_neighborhood(self, entity, flag = True):
        neighborhood = {}
        if isinstance(entity, str):
            if flag:
                idx = self.entity2id[entity]
            else:
                if '(' in entity:
                    entity = entity.split('(')[0].strip()
                idx = self.entity2id_[entity]
        else:
            idx = entity
        if str(idx) not in self.entity_kg:
            return neighborhood
        for relation_and_tail in self.entity_kg[str(idx)]:
            nei_idx = relation_and_tail[1]
            relation_idx = relation_and_tail[0]
            relation = self.entity2name(self.id2relation[relation_idx])
            try:
                nei_entity = self.id2entity[nei_idx]
                if relation in ['starring', 'writer', 'productionCompany', 'director', 'musicComposer']:#"self loop",
                    neighborhood[nei_entity] = relation
            except:
                pass
        return neighborhood

    def _process_two_hot_neighborhood_mv(self):
        self.neighthood_mv = {}
        for entity in tqdm(self.entity2id.values()):
            if str(entity) not in self.entity_kg:
                self.neighthood_mv[entity] = []
                continue
            one_hot_nei = self.neighborhood[entity]
            two_hot_nei = []
            for e in one_hot_nei:
                two_hot_nei.extend(self.neighborhood[e])
            two_hot_nei = list(set(two_hot_nei+one_hot_nei))
            nei_mv = []
            for e in two_hot_nei:
                if e in self.item_ids:
                    nei_mv.append(e)
            self.neighthood_mv[entity] = nei_mv

    def get_nei_mv(self, entity):
        if isinstance(entity, str):
            idx = self.entity2id[entity]
        else:
            idx = entity
        return self.neighthood_mv[idx]


    def get_movie_plot(self, movie):
        if isinstance(movie, int):
            movie = self.id2entity[movie]
        movie = self.entity2name(movie)
        plot = ""
        if movie in self.movie2plot.keys():
            plot = self.movie2plot[movie]["plot"]
        return plot
