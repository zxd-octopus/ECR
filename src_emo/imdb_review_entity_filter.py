import json
import re
import string
import html
from tqdm.auto import tqdm
from collections import defaultdict, Counter
from dataset_dbpedia import DBpedia

# filter the entities useful for review generation


def process(kg, data_file="data/redial/movie_reviews_filted_0.1_confi.json", out_file = "data/redial/movie_reviews_entities_filtered.json"):
    min_len = 9999
    with open('data/redial/stop_words.txt', 'r', encoding='utf-8') as f:
        stop_words = f.readlines()
        stop_words = [word.strip('\n') for word in stop_words]
    punctuation_string = string.punctuation
    cnt = defaultdict(int)
    relation_cnt = defaultdict(int)
    mv2e = {}
    total_cnt = Counter()
    e2mv = defaultdict(int)
    todo_e = set()
    with open(data_file, 'r', encoding='utf-8') as fin:
        reviews = json.load(fin)
        for mv_name, review_list in tqdm(reviews.items()):
            e_cnt = Counter()
            for review in review_list:
                content_e = review["content_e_0.1"]
                e = content_e #+ title_e
                e = list(sorted(e, key=lambda e: e[1]))
                e.reverse()
                nei_e = kg.get_one_hop_neighborhood(mv_name, False)
                e_relation = []
                e_review = []
                e_relation_name_set = []
                for e_ in e:
                    e_name = kg.entity2name(e_[0], True)
                    flag = False
                    for e_relation_name in nei_e.keys():
                        if e_name in e_relation_name:
                            if e_relation_name not in e_relation_name_set:
                                e_relation_name_set.append(e_relation_name)
                                e_relation.append([e_relation_name, nei_e[e_relation_name]])
                                relation_cnt[nei_e[e_relation_name]] += 1
                            flag = True
                    if flag == False:
                        if e_[0] not in e_review:
                            if e_name.lower() not in stop_words:
                                if mv_name.split('(')[0].strip() not in e_name:
                                    e_review.append(e_[0])

                review['e_review'] = e_review
                review['e_relation'] = e_relation
                for name in ['title_e', 'content_e', 'title_e_0.4', 'content_e_0.4', 'title_e_0.3', 'content_e_0.3', 'title_e_0.1', 'content_e_0.1']:
                    review.pop(name)
                e = review["e_review"]
                e_cnt.update(e)
                total_cnt.update(e)
                # cnt[len(e)] += 1
            mv2e[mv_name.split('(')[0].strip()] = sorted(dict(e_cnt).items(), key=lambda d: d[1],reverse = True)
            for e in dict(e_cnt).keys():
                e2mv[e] += 1
    total_cnt = sorted(dict(total_cnt).items(), key=lambda d: d[1],reverse = True) # 3418/72327>100  9500/72327>25
    eset = []
    for i in total_cnt:
        if i[1] < 20:
            break
        if e2mv[i[0]] > 1970:
            continue
        if len(kg.entity2name(i[0], True))<= 5:
            continue
        eset.append(i[0])


    for mv_name in tqdm(mv2e.keys()):
        thredshold = 2
        mv_e_list = []
        for mv_e in mv2e[mv_name]:
            if mv_e[0] not in eset:
                continue
            if mv_e[1] < thredshold:
                continue
            mv_e_list.append(mv_e)

        if len(mv_e_list) == 0:
            for mv_e in mv2e[mv_name]:
                if mv_e[0] not in eset:
                    continue
                mv_e_list.append(mv_e)
                break

        if len(mv_e_list)==0 and len(mv2e[mv_name]) != 0:
            mv_e_list.append(mv2e[mv_name][0])

        mv2e[mv_name] = mv_e_list
        cnt[len(mv_e_list)] += 1
    cnt = sorted(cnt.items(), key=lambda d: d[0])
    with open('data/redial/mv2e_review.json', 'w', encoding='utf-8') as f:
        json.dump(mv2e, f, ensure_ascii=False)
    with open('data/redial/review_e_filted.json', 'w', encoding='utf-8') as f:
        json.dump(eset, f, ensure_ascii=False)
    print("here")

    ef_cnt = defaultdict(int)
    for mv_name, review_list in tqdm(reviews.items()):
        mv2e_ = mv2e[mv_name.split('(')[0].strip()]
        mv2e_ = [e[0] for e in mv2e_]
        for review in review_list:
            e = review["e_review"]
            e_filtered = []
            for e_ in e:
                if e_ in mv2e_:
                    e_filtered.append(e_)
            if len(e_filtered) == 0 and len(e) != 0:
                e_filtered.append(e[0])
            review['e_review'] = e_filtered
            ef_cnt[len(e_filtered)] += 1

    ef_cnt = sorted(ef_cnt.items(), key=lambda d: d[0])

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(reviews, f, ensure_ascii=False)


if __name__ == '__main__':
    kg = DBpedia("redial_gen")
    process(kg)

