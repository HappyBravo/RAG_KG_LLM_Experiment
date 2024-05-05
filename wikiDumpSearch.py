# IMPORTING
from wikiDump_cleaner import Cleaner
import bz2, os, re, sys, json, pickle
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from tqdm import tqdm
from difflib import SequenceMatcher
# from joblib import Parallel, delayed 


class offline_Wiki():
    def __init__(self, 
                 wiki_index_file = None,
                 wikiDump_bz2_file = None,
                 index_folder = None,
                 verbose = False
                 ):
        self.verbose = verbose
        self.wiki_index_file = wiki_index_file
        self.wikiDump_bz2_file = wikiDump_bz2_file
        
        self.prefixx = "index_"
        self.suffixx = ".p" 
        self.files_index = []
        self.index_keys = []
        self.index_folder = index_folder

        if self.index_folder:
            self.files_index = os.listdir(index_folder)
            self.index_keys = sorted([".".join(os.path.basename(i).split(".")[:-1]).split("_")[1] for i in self.files_index], key=lambda x: x.lower())

        else:
            _choice = input("Wiki index folder for is not provided. Do you want to create index ? (1/0) : ").strip()
            if _choice in "10" and len(_choice) == 1 :
                if int(_choice):
                    self.index_folder = self.index_maker(wiki_index_file, wikiDump_bz2_file, verbose=self.verbose)
                    self.files_index = os.listdir(self.index_folder)
                    self.index_keys = sorted([".".join(os.path.basename(i).split(".")[:-1]).split("_")[1] for i in self.files_index], key=lambda x: x.lower())
                else:
                    print("Index folder not made ...")
    ###
    ### --- INDEX MAKER ---
    ###
                    
    def get_start_bytes_list(self, txt_wiki_dump_index_path, bz2_wiki_dump_path, verbose = False):
        start_bytes = []
 
        with open(txt_wiki_dump_index_path, "r", encoding="utf-8") as f:
            index_file = f.readlines()
        
        if verbose:
            start_bytes = [int(x.split(":")[0]) for x in tqdm(index_file)] # <--- SIMPLY USING A for LOOP IS VERY VERY FAST (7 SEC)
            # start_bytes = Parallel(n_jobs=N_JOB_COUNT)(delayed(get_start_bytes_list_helper)(x) for x in tqdm(index_file)) # <--- USING Parallel FINISHES IN >3 MINS 😮
        else:
            start_bytes = [int(x.split(":")[0]) for x in index_file]

        # to deduplicate the list
        start_bytes = list(set(start_bytes))  #
        
        # but we want them in a specific order

        file_size = os.path.getsize(bz2_wiki_dump_path)
        start_bytes.append(file_size + 1)

        start_bytes.sort()

        if verbose:
            print(f"GOT {len(start_bytes)} START BYTES")
        
        return start_bytes
    
    def acceptableWord(self, word, verbose = False):
        flag = False
    
        if verbose:
            print(f"Checking {word}", end="\r")
    
        if any((i in word) for i in ["File:", 
                                    "Template:", 
                                    "Wikipedia:", 
                                    "Category:", 
                                    "Help:", 
                                    "Portal:",
                                    "MediaWiki:",
                                    "Draft:",
                                    "Module:"]):
            return flag
        
        if any([word.endswith(ext) for ext in [".jpg",
                                                ".png", 
                                                ".gif", 
                                                ".zip", 
                                                ".ogg", 
                                                ".mp3", 
                                                ".mp4", 
                                                ".webp"]]):
            return flag
        
        flag = True
    
        return flag

    def clean_filename(self, filename):
        # Remove invalid characters for filenames
        return re.sub(r'[^\w\-_.() ]', ' ', filename)
    
    def save_pickle(self, dataa, filename):
        with open(filename, "wb") as f:
            pickle.dump(dataa, f)

    def store_dictionary_in_bins(self, word_dictionary, binsize=10000, 
                                index_folder ="./indexes/", verbose = False):
        if not os.path.exists(index_folder):
            os.makedirs(index_folder)

        sorted_keys = sorted(word_dictionary.keys())
        # sorted_keys = sorted(word_dictionary.keys(), key=lambda x: x.lower())  # Sort keys case-insensitively
        num_bins = len(sorted_keys) // binsize + (1 if len(sorted_keys) % binsize != 0 else 0)

        for i in tqdm(range(num_bins)):
            start_idx = i * binsize
            end_idx = min((i + 1) * binsize, len(sorted_keys))
            bin_keys = sorted_keys[start_idx:end_idx]
            bin_data = {key: (word_dictionary[key]) for key in bin_keys}

            first_key = bin_keys[0]
            filename = f"index_{self.clean_filename(first_key).strip()}.p"
            # Check if the cleaned filename is less than 3 characters
            if len(filename) < 8+2 and i > 0:
                # Try using the second key in bin_keys as the filename
                # if len(bin_keys) > 1:
                #     second_key = bin_keys[1]
                #     cleaned_filename = clean_filename(f"index_{second_key}.json")
            # if len(filename) < 8+3:
                print(i, filename, first_key , len(filename))
                # print(cleaned_filename, second_key , len(cleaned_filename))
                continue

            file_path = index_folder+filename
            self.save_pickle(bin_data, file_path)
            # # with open(file_path, 'w') as f:
            # #     json.dump(bin_data, f, indent=3)
            if verbose:
                print(f"Stored {len(bin_data)} elements in {filename}")

    def index_maker(self, index_file, wikiDump, index_folder = "./indexes/", verbose = False):
        if not index_file:
            index_file = self.wiki_index_file
        
        if not wikiDump :
            wikiDump = self.wikiDump_bz2_file
        
        # wiki_bz2_file_size = os.path.getsize(wikiDump)
        
        start_byte_list = self.get_start_bytes_list(index_file, wikiDump, verbose=verbose)
        
        if verbose:
            print("making start - end bytes list")
        start_end_list = [(start_byte_list[i], start_byte_list[i+1]) for i in range(len(start_byte_list)-1)]

        start_byte_list.clear() # EMPTYING MEMEORY
        
        if verbose:
            print("start - end bytes list made")

        start_end_dict = dict(start_end_list)

        word_start_end_dict = {}       
        if verbose : 
            print(f"Opening {index_file}.")

        with open(index_file, "r", encoding = "utf-8") as f:
            index_file = f.readlines()

        if verbose:
            print(f"length of index file : {len(index_file)}")

        for line in tqdm(index_file):
            start_byte, _idk, word = line.split(":")[0], line.split(":")[1], ":".join(line.split(":")[2:]).strip() 
            start_byte = int(start_byte)
            # word = ":".join(word)
            # if word 
            # print(f"{start_byte}, {start_end_dict[start_byte]} : {word}")
            if not self.acceptableWord(word):
                # print(word)
                continue
            word_start_end_dict[word] = (start_byte, start_end_dict[start_byte])
            # input()
            # start_byte = 
        # start_bytes = [int(x.split(":")[0]) for x in tqdm(index_file)] # <--- SIMPLY USING A for LOOP IS VERY VERY FAST (7 SEC)
        if verbose:
            print(f"Length of word - (start, end) dict is : {len(word_start_end_dict)}")

        index_file.clear() # EMPTYING MEMEORY

        if verbose:
            print("Making bins")
        self.store_dictionary_in_bins(word_start_end_dict, binsize=20000, index_folder=index_folder) # STORING IN BINS
        
        if verbose:
            print("Bins made")
        
        return index_folder

    ###
    ### --- INDEX READER ---
    ###

    def load_pickle(self, filename):
        res = None
        with open(filename, "rb") as f:
            res = pickle.load(f)
        return res
    
    ### ----- SIMILARITY FINDING FUNCTIONS | STARTS -----

    def search_closest_words(self, keywords, word):
        word_lower = word.lower()  # Convert search word to lowercase
        start = 0
        end = len(keywords) - 1
        closest_words = []

        while start <= end:
            mid = (start + end) // 2

            # Convert current keyword to lowercase for comparison
            keyword_mid_lower = keywords[mid].lower()

            # Check if the word falls between keywords[mid] and keywords[mid+1]
            if keyword_mid_lower < word_lower < keywords[mid + 1].lower():
                closest_words.append(keywords[mid])
                closest_words.append(keywords[mid + 1])
                break
            elif word_lower < keyword_mid_lower:
                end = mid - 1
            else:
                start = mid + 1

        return closest_words

    def jaccard_similarity_word(self, s1, s2):
        set1 = set(s1.lower())  # Convert s1 to lowercase
        set2 = set(s2.lower())  # Convert s2 to lowercase
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def jaccard_similarity(self, str1, str2):
    # Convert input strings to sets of words
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        # similarity = intersection / union
        
        # return similarity
        return intersection / union if union != 0 else 0

    def weighted_jaccard_similarity(self, s1, s2):
        set1 = set(s1.lower().split())  # Convert to lowercase and split into words
        set2 = set(s2.lower().split())
        # intersection_weight = sum(min(list(set1).count(w), list(set2).count(w)) for w in set1.intersection(set2))
        # union_weight = sum(max(list(set1).count(w), list(set2).count(w)) for w in set1.union(set2))
        intersection_weight = len(set1.intersection(set2))
        union_weight = len(set1.union(set2))
        return intersection_weight / union_weight if union_weight != 0 else 0
        
    def is_fuzz_similar(self, string1, string2, threshold = 80, verbose = False):
        ratioo = fuzz.ratio(string1, string2)
        if verbose:
            print(f"Fuzzy similarity between {string1} and {string2} is {ratioo}")
        if ratioo >= threshold:
            return True 
        return False 

    ### ----- SIMILARITY FINDING FUNCTIONS | ENDS -----

    def find_similar_keys(self, word, dictionary, threshold=0.5, verbose=False):
        similar_keys = []

        if verbose:
            print(f"Finding {word}...")

        word_lower = word.lower()  # Convert word to lowercase

        for key in dictionary:
            # similarity = fuzz.ratio(word_lower, key)
            # similarity = self.jaccard_similarity_word(word_lower, key)  # Use lowercase word for comparison
            similarity = self.jaccard_similarity(word_lower, key)  # Use lowercase word for comparison
            # similarity = weighted_jaccard_similarity(word_lower, key)  # Using weighted Jaccard
            if similarity > threshold:
                similar_keys.append((key, similarity))

        similar_keys.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity
        similar_keys = [key[0] for key in similar_keys]  # Extract keys only

        return similar_keys if similar_keys else None

    def fetch_word_from_list(self, target_word, similar_words, threshold = 0.7): 
        for word in similar_words:
            if word == target_word:
                return word
        for word in similar_words:    
            if word.lower() == target_word.lower():
                return word
        
        similarr = []
        word_lower = word.lower()
        for key in similar_words:
            # _similarity = fuzz.ratio(word_lower, key)
            _similarity = self.jaccard_similarity(word_lower, key)  # Use lowercase word for comparison

            # _similarity = self.weighted_jaccard_similarity(word_lower, key)  # Use lowercase word for comparison
            # similarity = weighted_jaccard_similarity(word_lower, key)  # Using weighted Jaccard
            if _similarity > threshold:
                similarr.append((key, _similarity))

        similarr.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity
        similarr = [key[0] for key in similarr]  # Extract keys only
        
        return similarr[0] if similarr else None
        # return find_most_similar_word(target_word, similar_words, max_similarity=threshold)  
        # return None

    def find_most_similar_word(self, query, keywords, max_similarity = 0.7):
        # max_similarity = 0
        most_similar_word = None

        # Iterate through the keywords and find the most similar one to the query
        for keyword in keywords:
            similarity = SequenceMatcher(None, query, keyword).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_word = keyword

        return most_similar_word


    def page_cleaner(self, page_body, summaryOnly = False):
        cleaner = Cleaner()
        cleaned_page = cleaner.clean_text(page_body)
        cleaned_text, links = cleaner.build_links(cleaned_page)

        if summaryOnly:
            cleaned_text = cleaned_text.split("==")[0] # THE FIRST PARAGRAPH/SECTION HAS THE SUMMARY

        return cleaned_text, links

    def extract_cleaned_page(self, page_soup, summaryOnly = False, verbose = False, wantLinks = False, wikibaseurl = "https://en.wikipedia.org/wiki/"):
        """
        RETURNS THE PAGE TITLE, URL, AND PAGE CONTENT
        YOU MAY WANT TO CHANGE THIS ACCORDING IT YOUR NEED
        
        """
        # print(page_soup.find("title"))
        # input()
        page_title = page_soup.find("title").text
        page_body = page_soup.find("text").text
        page_redirect = page_soup.find("redirect") 
        page_url = wikibaseurl+page_title.replace(' ', '_')

        if page_redirect:
            page_redirect = page_redirect['title']
            page_url = wikibaseurl+page_redirect.replace(' ', '_')
            # page_title = page_redirect.replace(' ', '_')
            
        page_body, _ = self.page_cleaner(page_body, 
                                    summaryOnly=summaryOnly)
    
        if verbose:
            print(f"Page cleaning done... \nGot Title : {page_title}, \nCleaned page body : {page_body} \nPage url : {page_url}, {'and Links : {_}'*wantLinks}\n")
        
        returning = [page_title, page_url, page_body]

        if wantLinks:
            returning.append(_)

        return returning

    def extract_pages(self, page_xml):
        soup = BeautifulSoup(page_xml, "lxml")
        pages = soup.find_all("page")
        return pages
    
    # def retrieve_text(self, title, offset):
    def decompress_xml2(self, title, offset):

        '''
        TAKEN FROM : "https://gerrit.wikimedia.org/r/plugins/gitiles/operations/dumps/+/ariel/toys/bz2multistream/wikiarticles.py"

        retrieve the page text for a given title from the xml file
        this does decompression of a bz2 stream so it's more expsive than
        other parts of this class
        arguments:
        title  -- the page title, with spaces and not underscores, case sensitive
        offset -- the offset in bytes to the bz2 stream in the xml file which contains
                  the page text
        returns the page text or None if no such page was found
        '''
        # self.xml_fd.seek(offset)
        with open(self.wikiDump_bz2_file, "rb") as f:
            f.seek(offset)

            unzipper = bz2.BZ2Decompressor()
            out = None
            found = False
            try:
                # block = self.xml_fd.read(262144)
                block = f.read(262144)

                out = unzipper.decompress(block).decode()
            # hope we got enough back to have the page text
            except:
                raise
            # format of the contents (and there are multiple pages per stream):
            #   <page>
            #   <title>AccessibleComputing</title>
            #   <ns>0</ns>
            #   <id>10</id>
            # ...
            #   </page>
            title_regex = re.compile(r"<page>(\s*)<title>%s(\s*)</title>" % re.escape(title))
            while not found:
                match = title_regex.search(out)
                if match:
                    found = True
                    text = out[match.start():]
                    if self.verbose:
                        sys.stderr.write("Found page title, first 600 characters: %s\n" % text[:600])
                    break
                # we could have a part of the regex at the end of the string, so...
                if len(out) > 40 + len(title):  # length of the above plus extra whitespace
                    out = out[-1 * (40 + len(title)):]
                try:
                    # block = self.xml_fd.read(262144)
                    block = f.read(262144)

                except:
                    # reached end of file (normal case) or
                    # something really broken (other cases)
                    break
                try:
                    out = out + unzipper.decompress(block).decode()
                except EOFError:
                    # reached end of bz2 stream
                    # EOFError  means we have some data after end of stream, don't care
                    pass
            if not found:
                return None
            out = text
            found = False
            text = ""
            while not found:
                ind = out.find("</page>")
                if ind != -1:
                    found = True
                    if self.verbose:
                        sys.stderr.write("Found end page tag\n")
                    text = text + out[:ind + len("</page>")]
                    break
                # we could have part of the end page tag at the end of the string
                text = text + out[:-1 * len("</page>") - 1]
                out = out[-1 * len("</page>"):]
                try:
                    # block = self.xml_fd.read(262144)
                    block = f.read(262144)

                except:
                    # reached end of file (normal case) or
                    # something really broken (other cases)
                    break
                try:
                    out = out + unzipper.decompress(block).decode()
                except EOFError:
                    # reached end of bz2 stream
                    # EOFError  means we have some data after end of stream, don't care
                    pass
            # if not found this can be partial text. should we return it? no
            if not found:
                if self.verbose:
                    sys.stderr.write("Found partial text but no end page tag. Text follows:\n")
                    sys.stderr.write(text)
                    sys.stderr.write("\n")
                text = None
            return text

    def decompress_xml(self, bz2_wiki_dump_path, start_byte, end_byte, verbose = False):
        decomp = bz2.BZ2Decompressor()
        with open(bz2_wiki_dump_path, 'rb') as f:
            f.seek(start_byte)
            block_size = end_byte - start_byte - 1
            print(block_size)
            input()
            # readback = f.read(end_byte - start_byte - 1)
            readback = f.read(max(256*1024, block_size) + 256*1024)

            page_xml = decomp.decompress(readback).decode()

            pages = self.extract_pages(page_xml)
            
            if verbose:
                print(f"FOUND : {len(pages)} PAGES BETWEEN {start_byte} BYTE AND {end_byte} BYTE.")
            
        return pages 
    
    def word_match(self, word, verbose = False, summaryOnly = True):
        near_words = self.search_closest_words(self.index_keys, word)
        if verbose:
            print(f"near words similar to {word} : {near_words}")
        near_words_file_path = [self.index_folder + self.prefixx + word + self.suffixx for word in near_words]
        
        to_search_into = {}
        [to_search_into.update(self.load_pickle(file_pathh)) for file_pathh in near_words_file_path]
        similar_keywords = self.find_similar_keys(word, to_search_into)
        # similar_keywords = self.find_most_similar_keys(word, to_search_into)

        if verbose:
            print(f"Similar keywords found in Wiki : {similar_keywords}")
        
        if not similar_keywords:
            if verbose:
                print("No similar keyword found !!!")
            return None
  
        wanted = self.fetch_word_from_list(word, similar_keywords)
        
        if not wanted:
            wanted = similar_keywords[0]
        
        if verbose:
            print(f"Wanted : {wanted}")

        _start, _end = to_search_into[wanted]
        
        if verbose:
            print(f"Byte start : {_start}, Byte end : {_end}")
        
        # decompressed_pages = self.decompress_xml(self.wikiDump_bz2_file, _start, _end)
        page_xml = self.decompress_xml2(wanted, _start)
        decompressed_pages = BeautifulSoup(page_xml, "lxml").find_all("page")

        # print(decompressed_pages)
        # input()


        for page_xml in decompressed_pages:
            # print(page_xml)
            _page_title,_page_url, _page_summary = "", "", ""
            _page_title, _page_url, _page_summary = self.extract_cleaned_page(page_xml, summaryOnly=summaryOnly, verbose=verbose)
            if self.is_fuzz_similar(wanted, _page_title, threshold=90,verbose=verbose): 
                # offline_dict[_page_title] = {'title' : _page_title, 
                #                             'url' : _page_url, 
                #                             'summary' : _page_summary}
                return {'title' : _page_title, 
                        'url' : _page_url, 
                        'summary' : _page_summary}
        return None
        
                

if __name__ =="__main__":
    print("Running")
    WIKI_INDEX_FILE = "D://WikiDump/enwiki-20240220-pages-articles-multistream-index.txt/enwiki-20240220-pages-articles-multistream-index.txt"
    WIKI_BZ2_FILE = "D://WikiDump/enwiki-20240220-pages-articles-multistream.xml.bz2"

    INDEX_FOLDER = "../indexes/"

    _wiki = offline_Wiki(wiki_index_file= WIKI_INDEX_FILE,
                         wikiDump_bz2_file=WIKI_BZ2_FILE,
                         index_folder=INDEX_FOLDER,
                         verbose=0)
    
    wiki_result = _wiki.word_match("tiger", summaryOnly=True, verbose=0)

    print(wiki_result)