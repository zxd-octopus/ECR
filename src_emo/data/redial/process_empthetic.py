import json
import re
import sys
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
sys.path.append("../..")
from collections import Counter

def clean_and_tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def calculate_repetition_rate(text):
    words = clean_and_tokenize(text)
    word_counts = Counter(words)
    repetitions = sum(count for word, count in word_counts.items() if count > 1)
    total_words = len(words)
    if total_words == 0:
        return 0
    return repetitions / total_words


def filter_llama2(data_file, part):
    words = ["I","me","my","mine"]
    data_file = data_file.format(part)
    with open(data_file, 'r', encoding='utf-8') as fin:
        reviews_filtered = []
        reviews = json.load(fin)
        for review in reviews:
            content = review['content']
            dict_i = {}
            keys = content.split()
            for key in keys:
                if key in dict_i.keys():
                    dict_i[key] = dict_i[key] + 1
                else:
                    dict_i[key] = 1

            for word in words:
                if word not in keys:
                    dict_i[word] = 0
            sum = 0
            for word in words:
                sum+= dict_i[word]
            if sum< 4:
                continue
            reviews_filtered.append(review)
        print(len(reviews_filtered))
        with open("llama2_{}_filtered.json".format(part), 'w', encoding='utf-8') as fout:
            json.dump(reviews_filtered, fout, ensure_ascii=False)

# if llama== True: out_file = "movie_reviews_processed_{}_llama.json
def process(llama = False, data_file="movie_reviews_entities_filtered.json", out_file = "movie_reviews_processed_{}.json"):
    if llama:
        max_len = 300
    else:
        max_len = 150
    samples = []
    with open(data_file, 'r', encoding='utf-8') as fin:
        reviews = json.load(fin)
        for mv_name, review_list in tqdm(reviews.items()):
            for review in review_list:
                title = review["title"]
                votes = review["votes"]
                content = review['content'][0]
                if calculate_repetition_rate(content) > 0.5:
                    continue
                sent_list_ = re.split("([.?!])", content)  # content.split('.')
                sent_list = []
                for i in sent_list_:
                    flag = False
                    for j in ['review', '10', 'Review', 'REVIEW']:
                        if j in i:
                            flag = True
                    if flag == False:
                        sent_list.append(i)
                content = ''.join(sent_list)
                if votes < 1 or len(content.split())< 20: # votes>0
                    continue

                if llama:
                    if votes < 5 or len(content.split())< 120: # llama
                        continue

                if len(content.split(" ")) <= max_len:
                    review['content'] = content
                else:
                    sent_list = re.split("([.?!])", content) #content.split('.')
                    if len(sent_list) == 1:
                        review['content'] = content
                    else:
                        sent_list = sent_list[: -1]
                        font_list = []
                        font_len = 0
                        for i in sent_list:
                            font_len += len(i.split(" "))
                            if font_len > max_len/2:
                                break
                            font_list.append(i)
                        if len(font_list) == 0:
                            font_list.append(sent_list[0])
                        font_content = ''.join(font_list)

                        back_list = []
                        back_len = 0
                        for i in sent_list[::-1]:
                            back_len += len(i.split(" "))
                            if back_len > max_len/2:
                                break
                            back_list.append(i)
                        if len(back_list) == 0:
                            back_list.append(sent_list[-1])
                        back_content = ''.join(back_list[::-1])
                        review['content'] = font_content +  back_content
                content = review['content']

                sample = {
                    'title' : title,
                    'content': content,
                    'movie_name': mv_name,
                    'e_review': review['e_review'],
                    'e_relation': review['e_relation']
                }
                samples.append(sample)
        train_processed_samples, valid_processed_samples = train_test_split(samples, train_size=0.99,shuffle=True)
        with open(out_file.format("valid"), 'w', encoding='utf-8') as fout:
            json.dump(valid_processed_samples, fout, ensure_ascii=False)
        with open(out_file.format("train"), 'w', encoding='utf-8') as fout:
            json.dump(train_processed_samples, fout, ensure_ascii=False)

if __name__ == '__main__':
    llama = False
    process(llama = llama)
    if llama:
        filter_llama2("movie_reviews_processed_{}_llama.json",part = "train")
        filter_llama2("movie_reviews_processed_{}_llama.json" ,part = "valid")

