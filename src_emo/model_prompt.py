import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv
# from einops import rearrange, reduce, repeat
from einops import rearrange,repeat

class KGPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,
        n_prefix_rec=None, n_prefix_conv=None, n_prefix_emo=None,
        n_emotion = None,
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.n_prefix_emo = n_prefix_emo

        entity_hidden_size = hidden_size // 2
        emo_hidden_size = hidden_size // 16

        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        #self.emo_embeds = emo_embeds
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        # self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index = edge_index # !!?
        # self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.edge_type = edge_type

        self.emo_embeds = nn.Parameter(torch.empty(n_emotion + 1, hidden_size))
        # self.emo_embeds = emo_embeds
        stdv = math.sqrt(6.0 / (self.emo_embeds.size(-2) + self.emo_embeds.size(-1)))
        self.emo_embeds.data.uniform_(-stdv, stdv)

        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)


        self.emo_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size ),
        )
        self.emo_proj2 = nn.Linear(token_hidden_size , emo_hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        #fuse entities representation and emotion reoresentation
        self.emo_fuse_proj1 = nn.Sequential(
            nn.Linear(hidden_size + emo_hidden_size, hidden_size ),
            nn.ReLU(),
            nn.Linear(hidden_size , hidden_size + emo_hidden_size ),
        )
        self.emo_fuse_proj2 = nn.Linear(hidden_size + emo_hidden_size , hidden_size)
        self.emo_fuse_proj3 = nn.Linear(hidden_size , hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        #prompt for emotion prediction
        if self.n_prefix_emo is not None:
            self.emo_prefix_embeds = nn.Parameter(torch.empty(n_prefix_emo, hidden_size))
            nn.init.normal_(self.emo_prefix_embeds)
            self.emo_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

        # self.attn_pdrop = attn_pdrop
        # if self.n_prefix_conv is not None:
        #     self.emotion_prefix = PrefixEncoder(self.hidden_size, self.attn_pdrop)

        self.copy_proj_1 = nn.Sequential(
            nn.Linear(emo_hidden_size, emo_hidden_size// 2),
            nn.ReLU(),
            nn.Linear(emo_hidden_size// 2, hidden_size),
        )
        self.copy_proj_2 = nn.Linear(hidden_size, 1)

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def set_emo_embed(self, emo_embeds: torch.Tensor):
        self.emo_embeds.data = emo_embeds

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index) + node_embeds # !!?
        #entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def get_emo_embeds(self):
        emo_embeds = self.emo_embeds
        emo_embeds = self.emo_proj1(emo_embeds) + emo_embeds
        emo_embeds = self.emo_proj2(emo_embeds)
        return emo_embeds

    def get_copy_sroce(self, emo_rep, nei_mv):
        """

        Args:
            emo_rep: (batch_size, entity_len, hidden_size)
            nei_mv:(batch_size, entity_len，n_entity)

        Returns: copy_score: (batch_size，n_entity)

        """
        copy_logit = self.copy_proj_1(emo_rep)
        copy_logit = self.copy_proj_2(copy_logit)
        copy_logit = copy_logit.repeat(1,1,nei_mv.size(-1))
        copy_logit = torch.einsum("ble,ble->ble", copy_logit, nei_mv)
        # copy_logit = copy_logit.masked_fill(nei_mv, 0.0)
        copy_logit = copy_logit.sum(-2)
        return  copy_logit

    def get_nei_rep(self, nei_mv):
        """
        Args:
            nei_mv:(batch_size, entity_len，n_entity)
        Returns: nei_rep: (batch_size，n_entity, hidden_size)
        """
        entity_embeds = self.get_entity_embeds() # (entity_len, hidden_size)
        entity_embeds = repeat(entity_embeds, 'l e -> b l e', b= nei_mv.size(0))
        nei_rep = torch.einsum("ble,beh->blh", nei_mv, entity_embeds)
        return nei_rep


    def forward(self, entity_ids=None, token_embeds=None, emotion_ids = None,
                emotion_probs = None, lastest_emotion_ids = None,last_emotion_probs = None,
                output_entity=False, nei_mvs = None,
                use_rec_prefix=False,use_conv_prefix=False, use_emo_prefix=False, use_copy = False, nei_mer = False):
        copy_logit = None
        #emotion_ids [bs,2]
        batch_size, entity_embeds, entity_len, token_len, emo_embeds, emotion_prefix = None, None, None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size)

        if emotion_ids is not None:
            batch_size, emotion_max_len = entity_ids.shape[:2]
            emo_embeds_org = self.get_emo_embeds()
            emo_embeds = emo_embeds_org[emotion_ids]  # (batch_size, entity_len, emotion_max_len, hidden_size)
            emo_embeds = torch.einsum("...lh,...l->...h", emo_embeds, emotion_probs)
            if lastest_emotion_ids is not None:
                lastest_embeds = emo_embeds_org[lastest_emotion_ids]
                lastest_embeds = torch.einsum("...lh,...l->...h", lastest_embeds, last_emotion_probs)
                # lastest_embeds = torch.mean(lastest_embeds, dim=-2)
            if use_copy:
                copy_logit = self.get_copy_sroce(emo_embeds, nei_mvs)
            if nei_mer:
                nei_rep = self.get_nei_rep(nei_mvs)
                emo_embeds = repeat(emo_embeds, 'b l h -> m b l h', m= 2)
                emo_embeds = rearrange(emo_embeds, 'm b l e -> b (m l) e')

        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, hidden_size)
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None : # pretrain,conv,rec

            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  # (batch_size, token_len, entity_len)
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds

                if emo_embeds is not None:
                    if nei_mer:
                        prompt_embeds = torch.cat((prompt_embeds, nei_rep), 1)
                        entity_len = entity_len*2

                    # prompt_embeds_ = torch.cat((prompt_embeds, emo_embeds), -1)
                    # prompt_embeds_ = self.emo_fuse_proj1(
                    #     prompt_embeds_)  # + entity_embeds  # (batch_size, token_len, hidden_size)
                    # prompt_embeds_ = self.emo_fuse_proj2(
                    #     prompt_embeds_) + prompt_embeds
                    # prompt_embeds = self.emo_fuse_proj3(
                    #     prompt_embeds_)
                    # 以下为无残差网络的写法
                    prompt_embeds = torch.cat((prompt_embeds, emo_embeds), -1)
                    prompt_embeds = self.emo_fuse_proj1(
                        prompt_embeds)  # + entity_embeds  # (batch_size, token_len, hidden_size)
                    prompt_embeds = self.emo_fuse_proj2(
                        prompt_embeds)
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec

        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)

            # # emotion prompt for conversation
            # if lastest_emotion_ids is not None:
            #     emotion_prefix = self.emotion_prefix(lastest_embeds)
            #     prefix_embeds = torch.cat([prefix_embeds,emotion_prefix], dim=1)
            #     n_prefix_conv = self.n_prefix_conv + emotion_prefix.shape[1]
            # else:
            #     n_prefix_conv = self.n_prefix_conv

            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv



        if self.n_prefix_emo is not None and use_emo_prefix:
            prefix_embeds = self.emo_prefix_proj(self.emo_prefix_embeds) + self.emo_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_emo

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        return prompt_embeds, copy_logit

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)
