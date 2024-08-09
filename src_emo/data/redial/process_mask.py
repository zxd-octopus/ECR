import json
import re

import html
from tqdm.auto import tqdm

movie_pattern = re.compile(r'@\d+')
sentiment_dict = {0:-1, 2:0, 1:1}

def process_utt(utt, movieid2name, replace_movieId, remove_movie=False):
    def convert(match):
        movieid = match.group(0)[1:]
        if movieid in movieid2name:
            if remove_movie:
                return '<movie>'
            movie_name = movieid2name[movieid]
            # movie_name = f'<soi>{movie_name}<eoi>'
            return movie_name
        else:
            return match.group(0)

    if replace_movieId:
        utt = re.sub(movie_pattern, convert, utt)
    utt = ' '.join(utt.split())
    utt = html.unescape(utt)

    return utt


def process(data_file, out_file, movie_set, movie_name_set = None):
    cnt = 0
    cnt_total = 0
    with open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            dialog = json.loads(line)
            if len(dialog['messages']) == 0:
                continue
            turns = []
            movieid2name = dialog['movieMentions']
            user_id, resp_id = dialog['initiatorWorkerId'], dialog['respondentWorkerId']
            context, resp = [], ''
            entity_list = []
            emo_list = []
            sentiment_list = []
            emo_probs_list = []
            messages = dialog['messages']
            turn_i = 0
            id2movie = dialog['movieMentions'] if len(dialog['movieMentions']) != 0 else {}
            movie2id = {name.replace(" ", "") if name is not None else '': id for id, name in id2movie.items()}
            Ilabel = dialog['initiatorQuestions'] if len(dialog['initiatorQuestions']) != 0 else {}

            while turn_i < len(messages):
                worker_id = messages[turn_i]['senderWorkerId']
                utt_turn = []
                entity_turn = []
                movie_turn = []
                movie_name_turn = []
                I_movie_sentiment_turn = []
                mask_utt_turn = []
                emo_turn = []
                emo_lastest = []
                emo_probs_lastest = []
                emo_prob_turn = []

                turn_j = turn_i
                while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == worker_id:
                    utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True, remove_movie=False)
                    utt_turn.append(utt)

                    mask_utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True, remove_movie=True)
                    mask_utt_turn.append(mask_utt)

                    entity_ids = [entity2id[entity] for entity in messages[turn_j]['entity'] if entity in entity2id]
                    entity_turn.extend(entity_ids)

                    movie_ids = [entity2id[movie] for movie in messages[turn_j]['movie'] if movie in entity2id]
                    movie_ids_s = [movie2id[movie.replace(" ", "")] for movie in messages[turn_j]['movie_name']]
                    I_movie_sentiment = []
                    for mid in movie_ids_s:
                        I_movie_sentiment.append(Ilabel[mid]["liked"] if mid in Ilabel.keys() else 2)

                    I_movie_sentiment = [sentiment_dict[i] for i in
                                         I_movie_sentiment]

                    movie_turn.extend(movie_ids)
                    movie_name_turn.extend(messages[turn_j]['movie_name'])
                    I_movie_sentiment_turn.extend(I_movie_sentiment)


                    turn_emotion = []
                    turn_emotion_probs = []
                    try:
                        turn_emotion_ = messages[turn_j]['emotion']
                        for emo, probs in turn_emotion_.items():
                            if probs > 0.1:
                                turn_emotion.append(emo)
                                turn_emotion_probs.append(probs)
                    except:
                        pass
                    emo_turn.extend(turn_emotion)
                    emo_prob_turn.extend(turn_emotion_probs)

                    turn_j += 1

                utt = ' '.join(utt_turn)
                mask_utt = ' '.join(mask_utt_turn)
                movie_sentiment_turn = I_movie_sentiment_turn

                role = "seeker" if worker_id == user_id else "recommender"
                turns.append({"utt": utt,"mask_utt":mask_utt, "role": role, "emo_turn":emo_turn, "emo_probs_turn":emo_prob_turn,
                              "entity_turn": entity_turn,"movie_turn": movie_turn,"movie_sentiment_turn":movie_sentiment_turn,"movie_name_turn":movie_name_turn})
                turn_i = turn_j

            for i in range(len(turns)):

                turn = turns[i]
                role = turn["role"]
                movie_sentiment_turn = turn['movie_sentiment_turn']

                next_turn = turns[i + 1] if i + 1 < len(turns) else None
                emotion = turn["emo_turn"] if role == "seeker" else next_turn[
                    "emo_turn"] if next_turn is not None else []
                emotion_porbs = turn["emo_probs_turn"] if role == "seeker" else next_turn[
                    "emo_probs_turn"] if next_turn is not None else []

                utt = turn["utt"]
                mask_utt = turn["mask_utt"]

                if role == "seeker":
                    context.append(utt)
                    entity_list.extend(turn["movie_turn"] + turn["entity_turn"])
                    emo_list.extend([emotion] * (len(turn["movie_turn"]) + len(turn["entity_turn"])))
                    emo_probs_list.extend([emotion_porbs] * (len(turn["movie_turn"]) + len(turn["entity_turn"])))
                    emo_lastest = turn["emo_turn"]
                    emo_probs_lastest = turn["emo_probs_turn"]
                else:
                    resp = utt

                    if role == "recommender" and len(context) == 0:
                        context.append('')

                    assert len(entity_list) == len(emo_list), "entity and emotion is not match"

                    sample = {
                        'role' : role,
                        'context': context,
                        'resp': mask_utt,
                        'rec': turn["movie_turn"],
                        'rec_weight': emotion if  len(emotion)>0 else ["neutral"],
                        'rec_weight_w': emotion_porbs if len(emotion_porbs)>0 else [1.0],
                        'sentiment_loss': movie_sentiment_turn,
                        'entity': entity_list,
                        'emotion_entity': emo_list,
                        'emotion_lastest': emo_lastest,
                        'emotion_probs_entity': emo_probs_list,
                        'emo_probs_lastest':emo_probs_lastest,
                        # 'sentiment_list': sentiment_list,
                    }
                    for item in turn["movie_turn"]:
                        if item in entity_list:
                            cnt += 1
                    cnt_total += len(turn["movie_turn"])

                    fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

                    context.append(resp)
                    entity_list.extend(turn["movie_turn"] + turn["entity_turn"])
                    emo_list.extend([emotion] * (len(turn["movie_turn"]) + len(turn["entity_turn"])))
                    emo_probs_list.extend([emotion_porbs] * (len(turn["movie_turn"]) + len(turn["entity_turn"])))
                    movie_set |= set(turn["movie_turn"])
                    if movie_name_set is not None:
                        movie_name_set |= set(turn["movie_name_turn"])

    print(cnt, cnt_total)
# def process_pretrain( data_file,  out_file):
#     for name in ['train','valid','test']:
#         with open(data_file.format(name), 'r') as fin, open(out_file.format(name), 'w', encoding='utf-8') as fout:
#             datas = json.load(fin)
#             for data in datas:
#                 sample = {}
#                 sample["context"] = [" ".join(utt) for utt in data["context"]]
#                 sample["resp"] = " ".join(data["target"])
#                 sample["emotion_lastest"] = data["emotion"]
#                 sample['emotion_entity'] = []
#                 sample['emo_probs_lastest'] = [1.0]
#                 sample['emotion_probs_entity'] = []
#                 fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    with open('entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)
    id2entity = {v: k for k, v in entity2id.items()}
    movie_set = set()
    movie_name_set = set()
    # with open('node2abs_link_clean.json', 'r', encoding='utf-8') as f:
    #     node2entity = json.load(f)

    process('valid_data_dbpedia_emo.jsonl', 'valid_data_processed.jsonl', movie_set)
    process('test_data_dbpedia_emo.jsonl', 'test_data_processed.jsonl', movie_set)
    process('train_data_dbpedia_emo.jsonl', 'train_data_processed.jsonl', movie_set, movie_name_set)


    # process_pretrain('emp_dataset_{}.json', 'emo_datasest_processed_{}.json')

    with open('movie_ids.json', 'w', encoding='utf-8') as f:
        json.dump(list(movie_set), f, ensure_ascii=False)
    with open('movie_name.json', 'w', encoding='utf-8') as f:
        json.dump(list(movie_name_set), f, ensure_ascii=False)
    print(f'#movie: {len(movie_set)}')
