import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--gen_file_prefix", type=str, required=False, default="rec.json")
args = parser.parse_args()
gen_file_prefix = args.gen_file_prefix
dataset = 'redial_gen'
gen_dataset = 'redial_rec'

gen_file_path = f"../../save/{gen_dataset}/{gen_file_prefix}"
gen_file = open(gen_file_path, encoding='utf-8')
gen_data = gen_file.readlines()
cnt = 0
for split in ['valid', 'test']:
    raw_file_path = f"{split}_data_processed.jsonl"
    raw_file = open(raw_file_path, encoding='utf-8')
    raw_data = raw_file.readlines()
    # print(len(raw_data))

    new_file_path = f'{split}_data_processed_merge.jsonl'
    new_file = open(new_file_path, 'w', encoding='utf-8')


    for raw in raw_data:
        raw = json.loads(raw)
        if len(raw['context']) == 1 and raw['context'][0] == '':
            raw['rec_gen'] = []
        else:
            rec_gen = []
            for i in range(len(raw['rec'])):
                gen = json.loads(gen_data[cnt])
                pred = gen['pred']
                rec_gen.append(pred)
                cnt += 1
            raw['rec_gen'] = rec_gen
        new_file.write(json.dumps(raw, ensure_ascii=False) + '\n')

assert cnt == len(gen_data)
