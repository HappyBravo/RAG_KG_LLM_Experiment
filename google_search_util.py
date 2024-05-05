# IMPORTS

import requests, re

from bs4 import BeautifulSoup
from googlesearch import search
from GoogleNews import GoogleNews
from googletrans import Translator

# from joblib import Parallel, delayed
from tqdm import tqdm

# import httpcore
# setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')

class GoogleUtil():
    def __init__(self):
        # self.translator = Translator()
        pass

    def clean_spaces_text(self, text):
        # Replace multiple spaces, tabs, and newline characters with a single space
        cleaned_text = re.sub(r'\s+', ' ', text)
        return cleaned_text.strip()  # Remove leading and trailing spaces

    def _search_google(self, query, number_of_results):
        urls = []
        for url in search(query, num_results=number_of_results):
            urls.append(url)
        return urls
    
    def search_google(self, query, number_of_results = 5, englishOnly = True):
        search_results = self._search_google(query=query, number_of_results=number_of_results)
        google_search_data = {}

        for url in search_results:
            article_text = self.fetch_articles(url)
            if article_text:
                _temp_text = self.clean_spaces_text(article_text)
                article_text = _temp_text
                
                # if englishOnly and self.contains_hindi(_temp_text):
                #     translated_text = self.translator.translate(_temp_text, dest="en")
                #     article_text = translated_text
                
                google_search_data[url] = article_text
        if englishOnly:
            return self.translate_hindi_to_english(google_search_data)
        return google_search_data
                    


    def fetch_articles(self, url_to_search): # FETCH THE CONTENT FROM THE WEBPAGE
        try:
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

            response = requests.get(url_to_search, 
                                    timeout=30,
                                    # timeout=10, # IF WE HAVE DECENT NET SPEED
                                    headers=headers,
                                    )
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text from webpage
                text_content = ' '.join([p.text for p in soup.find_all('p')])
                return text_content
            else:
                print("Failed to fetch webpage:", response.status_code)
                return None
        except Exception as e:
            print("An error occurred:", str(e))
            return None
        
        
    def _search_google_news(self, query,             # QUERY / KEYWORDS TO SEARCH
                            number_of_results = 5,   # NUMBER OF ARTICLES / NEWS TO FETCH
                            period = "7d",           # PAST DAYS
                            useGoogle = True):          
        news_urls = []
        googlenews = GoogleNews()
        
        # FILTERS AND PARAMETERS
        googlenews.set_lang("en")
        googlenews.set_period(period)
        googlenews.set_encode("utf-8")

        if useGoogle:
            googlenews.search(query)
        else:
            googlenews.get_news(query)

        results = googlenews.results()
        # print(googlenews.get_links())

        # input()

        # print(results)
        for i in range(min(len(results), number_of_results)):
            _link = results[i]['link']
            if "https://" not in _link[:10]:
                _link = "https://"+_link 
            news_urls.append(_link)
        googlenews.clear()
        return news_urls
    
    def search_google_news(self, query, number_of_results = 5, period = "7d", englishOnly = True, useGoogle = True):
        news_links = self._search_google_news(query=query, 
                                              number_of_results=number_of_results, 
                                              period=period,
                                              useGoogle=useGoogle)
        news_links = list(map(self.clean_links, news_links))
        # print(news_links)

        google_news_data = {}

        for url in news_links:
            article_text = self.fetch_articles(url)
            if article_text:
                _temp_text = self.clean_spaces_text(article_text)
                article_text = _temp_text
                
                # if englishOnly and self.contains_hindi(_temp_text):
                #     translated_text = self.translator.translate(_temp_text, dest="en")
                #     article_text = translated_text
                
                google_news_data[url] = article_text
        if englishOnly:
            return self.translate_hindi_to_english(google_news_data)
        return google_news_data


    def contains_hindi(self, text): # CHECK IF THE TEXT CONTAINS HINDI WORDS/CHARACTERS
        hindi_pattern = re.compile(r'[\u0900-\u097F]+')
        if hindi_pattern.search(text):
            print("contains hindi !")
            return True
        return False

    def clean_links(self, link):
        link = link.strip().split("&ved")[0] # IDK WHAT THIS "&ved" IS ... IT GOT ATTATCHED AUTOMATICALLY... DUE TO WHICH IT WAS GIVING ERROR 404 
        return link

    def translate_hindi_to_english(self, searched_data_dict):
        translator = Translator()
        for key, value in searched_data_dict.items():
            if self.contains_hindi(value):
                print(f"{key} - Contains Hindi")
                try:
                    translated_text = translator.translate(value, dest="en")
                    searched_data_dict[key] = translated_text.text
                except Exception as e:
                    print("Error translating. Error:", e)
                    searched_data_dict[key] = "" # ENGLISH ONLY !
        return searched_data_dict
    
    def fetch_data_from_relation(self, relations=[], number_of_results = 5, 
                                 useGoogleForNews = True):
        search_dict = {}
        news_dict = {}
        searched = []
        for _relation in relations:
            ent1, ent2 = _relation
            search_space = [f'"{ent1}"', f'"{ent2}"', f'"{ent1} {ent2}"']
            for _ele in search_space:
                if _ele in searched:
                    continue
                print(f"Searching for {_ele}...")
                searched.append(_ele)
                search_dict.update(self.search_google(_ele, number_of_results = number_of_results))
                news_dict.update(self.search_google_news(_ele, number_of_results=number_of_results, useGoogle=useGoogleForNews))
                # print(" "*(5 + len(_ele)), end = "\r")
            print()
        return search_dict, news_dict

if __name__ == "__main__":
    query = "West Bengal election sun"
    n = 10

    googleBaba = GoogleUtil()

    search_data = googleBaba.search_google(query, number_of_results= n)
    for i in search_data:
        print(">>> ", i)
        print(search_data[i])
    print()
    print("-----")
    print()
    news_data = googleBaba.search_google_news(query, number_of_results=n, useGoogle=True)
    for i in news_data:
        print(">>> ", i)
        print(news_data[i])
    print("len(news_data)", len(news_data))
