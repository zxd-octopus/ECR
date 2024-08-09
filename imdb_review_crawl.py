import imdb
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import time

ia = imdb.IMDb()

def get_imdb_id(mv_name):
    search_movie = ia.search_movie(mv_name)
    ID = search_movie[0].getID()
    return ID


BASE_URL = "https://www.imdb.com/title/{}/reviews?spoiler=hide&sort=helpfulnessScore&dir=desc&ratingFilter=10"
HEADERS = {
    'User-Agent': 'Mozilla/5.o (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.162 Safari/537.36'
}


def fetch_movie_reviews(movie_id):
    response = requests.get(BASE_URL.format(movie_id), headers=HEADERS)
    if response.status_code == 200:
        return response.content
    else:
        return None


def extract_reviews(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    reviews = []

    for review_div in soup.select('.imdb-user-review'):
        title = review_div.select_one('.title').get_text(strip=True)
        content = review_div.select_one('.content .text').get_text(strip=True)
        votes = int(review_div.select_one('.actions').get_text(strip=True).split()[0].replace(',',''))

        reviews.append({
            'title': title,
            'content': content,
            'votes': votes
        })

    return reviews


def display_reviews(mv_name):
    movie_id = 'tt'+str(get_imdb_id(mv_name))
    html_content = fetch_movie_reviews(movie_id)
    if not html_content:
        print("failed")
        return

    reviews = extract_reviews(html_content)# at most 25 reviews for one movie
    sorted_reviews = sorted(reviews, key=lambda x: x['votes'], reverse=True)
    return sorted_reviews

if __name__ == "__main__":
    while(1):
        cnt = 0
        miss = 0
        mv_name = None
        movie_review = None
        with open("src_emo/data/redial/movie_name.json", 'r', encoding='utf-8') as fin:
            mv_names = json.load(fin)
        try:
            with open('src_emo/data/redial/movie_reviews.json', 'r', encoding='utf-8') as fin:
                movie_review = json.load(fin)
        except:
            movie_review = {}

        mv_names_e = movie_review.keys()
        for mv_name in tqdm(mv_names):
            if mv_name not in mv_names_e:
                try:
                    sorted_reviews = display_reviews(mv_name)
                    movie_review[mv_name] = sorted_reviews
                    cnt += 1
                except:
                    print(mv_name)
                    miss += 1
                time.sleep(1)
            else:
                pass
            if cnt == 50:
                break
        print("total {} movies did not find reviews".format(miss))
        with open('src_emo/data/redial/movie_reviews.json', 'w', encoding='utf-8') as f:
            json.dump(movie_review, f, ensure_ascii=False)
        if mv_name == mv_names[-1]:
            print("all done!")
            break
