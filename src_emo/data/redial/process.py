import json
import re

import html
from tqdm.auto import tqdm

movie_pattern = re.compile(r'@\d+')


def process_utt(utt, movieid2name, replace_movieId):
    def convert(match):
        movieid = match.group(0)[1:]
        if movieid in movieid2name:
            movie_name = movieid2name[movieid]
            movie_name = ' '.join(movie_name.split())
            return movie_name
        else:
            return match.group(0)

    if replace_movieId:
        utt = re.sub(movie_pattern, convert, utt)
    utt = ' '.join(utt.split())
    utt = html.unescape(utt)

    return utt


def process(data_file, out_file, movie_set):
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
            emo_probs_list = []
            messages = dialog['messages']
            turn_i = 0
            while turn_i < len(messages):
                worker_id = messages[turn_i]['senderWorkerId']
                utt_turn = []
                entity_turn = []
                movie_turn = []
                emo_turn = []
                emo_prob_turn = []

                turn_j = turn_i
                while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == worker_id:
                    utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True)
                    utt_turn.append(utt)

                    entity_ids = [entity2id[entity] for entity in messages[turn_j]['entity'] if entity in entity2id]
                    entity_turn.extend(entity_ids)

                    movie_ids = [entity2id[movie] for movie in messages[turn_j]['movie'] if movie in entity2id]
                    movie_turn.extend(movie_ids)
                    turn_emotion = []
                    turn_emotion_probs = []
                    try:
                        turn_emotion_ = messages[turn_j]['emotion']
                        for emo, probs in turn_emotion_.items():
                            if probs>0.1:
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



                utt = ' '.join(utt_turn)

                role = "seeker" if worker_id == user_id else "recommender"
                #     context.append(utt)
                #     entity_list.append(entity_turn + movie_turn)
                # else:
                resp = utt
                turns.append({"resp":resp, "role":role, "emo_turn":emo_turn, "emo_probs_turn":emo_prob_turn,
                              "entity_turn":entity_turn, "movie_turn":movie_turn})
                turn_i = turn_j

            for i in range(len(turns)):
                turn = turns[i]
                next_turn = turns[i+1] if i+1<len(turns) else None

                role = turn["role"]
                resp = turn["resp"]

                emotion = turn["emo_turn"] if role == "seeker" else next_turn["emo_turn"]  if next_turn is not None else []
                emotion_porbs = turn["emo_probs_turn"] if role == "seeker" else next_turn["emo_probs_turn"]  if next_turn is not None else []

                if role == "recommender" and len(context) == 0:
                    context.append('')
                assert len(entity_list) == len(emo_list),"entity and emotion is not match"
                sample = {
                    'role' : role,
                    'context': context,
                    'resp': resp,
                    'rec': list(set(turn["movie_turn"] + turn["entity_turn"])),
                    'entity': entity_list,
                    'emotion_entity':emo_list,
                    'emotion_probs_entity':emo_probs_list,
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

                context.append(turn["resp"])
                entity_list.extend(turn["movie_turn"] + turn["entity_turn"])
                emo_list.extend([emotion]*(len(turn["movie_turn"])+len(turn["entity_turn"])))
                emo_probs_list.extend([emotion_porbs]*(len(turn["movie_turn"])+len(turn["entity_turn"])))
                movie_set |= set(turn["movie_turn"])



if __name__ == '__main__':
    with open('entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)
    item_set = set()

    process('valid_data_dbpedia_emo.jsonl', 'valid_data_processed.jsonl', item_set)
    process('test_data_dbpedia_emo.jsonl', 'test_data_processed.jsonl', item_set)
    process('train_data_dbpedia_emo.jsonl', 'train_data_processed.jsonl', item_set)

    with open('item_ids.json', 'w', encoding='utf-8') as f:
        json.dump(list(item_set), f, ensure_ascii=False)
    print(f'#item: {len(item_set)}')
