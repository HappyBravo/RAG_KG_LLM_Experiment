# IMPORT
import os, json, pickle, re, math, torch, multiprocessing, nltk, time
from tqdm import tqdm
import numpy as np

import wikipedia
from joblib import Parallel, delayed 
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import networkx as nx
from pyvis.network import Network
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# CLASSES
from wikiDumpSearch import offline_Wiki

from KB import KB

from NER import NER 

from RAG import RAG

from llama3 import llama3


# FOR GPU
def get_cpu_count():
    c = multiprocessing.cpu_count()
    return c 

def check_gpu():
    for i in range(torch.cuda.device_count()):
        device_name = f'cuda:{i}'
        print(f'{i} device name:{torch.cuda.get_device_name(torch.device(device_name))}')

def get_gpu():
    return [f'cuda:{i}' for i in range(torch.cuda.device_count())]

print(check_gpu())

print(get_gpu())

print(get_cpu_count())

N_JOB_COUNT = get_cpu_count()//2
N_JOB_COUNT = 1

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

# HELPER FUNCTIONS
def loadJSON(filepathh):
    _dataa = {} 
    if os.path.exists(filepathh):
        with open(filepathh, "r", encoding="utf-8") as _f:
            _dataa = json.load(_f)
    else:
        print(f"{filepathh} does not exists...\n") 
    return _dataa 

def loadTXT(filepathh):
    _dataa = ""
    if os.path.exists(filepathh):
        with open(filepathh, "r", encoding="utf-8") as _f:
            _dataa = _f.read()
    else:
        print(f"{filepathh} does not exists...\n") 
    return _dataa 

def loadFILE(filepathh = ""):
    if os.path.exists(filepathh):
        if filepathh.endswith(".txt"):
            return loadTXT(filepathh)
        elif filepathh.endswith(".json"):
            return loadJSON(filepathh)
        else:
            print("\n- Invalid File format 😐 !!!\n")
            return None
    else:
        print(f"{filepathh} does not exists...\n") 

def remove_garbage(text):
    # Remove garbage Unicode characters
    cleaned_text = text.encode().decode('unicode-escape')
    # Remove any remaining non-printable characters
    cleaned_text = re.sub(r'[^\x20-\x7E]', '', cleaned_text)
    return cleaned_text

def clean_sentence(sentence):
    # Remove extra white spaces
    cleaned_sentence = re.sub(r'\s+', ' ', sentence)
    # Remove unwanted characters except alphabets, numbers, punctuation marks, '@', '-', and '_'
    cleaned_sentence = re.sub(r'[^a-zA-Z0-9@#\-_.,?!\'" ]', '', cleaned_sentence)
    # Remove words containing '#' and 'pic.twitter.com'
    cleaned_sentence = ' '.join(word if '#' not in word and 'pic.twitter.com' not in word else ' ' for word in cleaned_sentence.split() )
    return cleaned_sentence.strip()

def clean_document(document):
    document = remove_garbage(document)
    # Tokenize the document into sentences
    sentences = sent_tokenize(document)
    # Clean each sentence
    cleaned_sentences = [clean_sentence(sentence) for sentence in sentences]
    return cleaned_sentences

if __name__ == "__main__":

    # LOADING WIKI
    WIKI_INDEX_FILE = "D://WikiDump/enwiki-20240220-pages-articles-multistream-index.txt/enwiki-20240220-pages-articles-multistream-index.txt"
    WIKI_BZ2_FILE = "D://WikiDump/enwiki-20240220-pages-articles-multistream.xml.bz2"

    INDEX_FOLDER = "./indexes/"

    offline_wikipedia = offline_Wiki(wiki_index_file=WIKI_INDEX_FILE,
                                    wikiDump_bz2_file=WIKI_BZ2_FILE, 
                                    index_folder=INDEX_FOLDER,
                                    verbose=False,)

    # testing wiki
    # print(offline_wikipedia.word_match("tigers", summaryOnly=False))
    # input()

    # test_words = ["sachin", "trump", "wedding", "white house", "modi", "apple", "windows 11"]
    # for test_word in test_words:
    #     print(offline_wikipedia.word_match(test_word, summaryOnly=False))
    
    # input("Press ENTER key to continue...")
    
    print("Loading NER models ... ")
    # Load model and tokenizer for NER
    ner_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    ner_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    ner_model.to(device)

    ner = NER(model=ner_model, tokenizer=ner_tokenizer, device = device)
    print("NER module initialized ... ")
    # TESING KB
    
    # input("Press ENTER key to continue...")
    
    # print("Testing KB and NER ... ")

    # statement = "The Sun rises in East direction. The Earth is smaller than the planet Jupiter. Tigers are from Cat family."
    # print("Statement -", statement)

    #######################################################
    # TAKEN FROM \9th Sem\Fake News Detection\knowledge_graphs\FNN_data\fakenewsnet_dataset\gossipcop\fake\gossipcop-2917215
    # LABEL = FAKE

    testing_texts = """Fans have always longed for \u201990s golden couple Brad Pitt and Jennifer Aniston to get back together \u2014 something Justin Theroux became keenly aware of two years ago when he found love notes Aniston had saved from the heartthrob, a source recalls in the new issue of Us Weekly.\n\nRelated: Jennifer Aniston and Justin Theroux: The Way They Were Oh, how sweet it is! Take a look back at Jennifer Aniston and Justin Theroux's romance -- from their engagement, sexy getaways, annual star-studded holiday parties and more!\n\n\u201cHe stumbled upon old Post-it notes Brad had written,\u201d a source tells Us. \u201cSweet little Post-its like, \u2018You looked nice tonight\u2019 or \u2018Miss you already.\u2019\u201d\n\nThe source explains that impact of those little notes was huge. \u201cJen assured him they weren\u2019t a big deal, but Justin wasn\u2019t thrilled \u2026 Justin had moments of insecurity like that.\u201d\n\nRelated: Jennifer Aniston and Justin Theroux Split: Revisit Their Sweetest Quotes About Each Other They once had it all! Jennifer Aniston and Justin Theroux have called it quits on their marriage after more than two years of marriage. The A-list couple confirmed their split in a statement to Us Weekly on Thursday, February 15. \u201cIn an effort to reduce any further speculation, we have decided to announce our separation. [\u2026]\n\nAs previously reported, Aniston, 49, and Theroux, 46, announced their separation on Thursday, February 15, via a joint statement: \u201cIn an effort to reduce any further speculation, we have decided to announce our separation. This decision was mutual and lovingly made at the end of last year. We are two best friends who have decided to part ways as a couple, but look forward to continuing our cherished friendship.\u201d\n\nThe duo, who started dating in May 2011, tied the knot in August 2015. While it was Theroux\u2019s first marriage, Aniston was married to Pitt, 54, from 2000 to 2005. He infamously moved on from the Friends alum with his Mr. and Mrs. Smith costar, Angelina Jolie. Pitt and Jolie, who share Maddox, 16, Pax, 14, Zahara, 12, Shiloh, 11 and 9-year-old twins Knox and Vivienne, called it quits in September 2016 after two years of marriage and 12 years together.\n\n\u201cJen has struggled with the perception that she is this pathetic woman after the divorce from Brad,\u201d the insider tells Us. \u201cIt played a role in her wanting to marry Justin.\u201d\n\nFor more on Aniston and Theroux\u2019s split, pick up the new issue of Us Weekly, on stands now!"""
    
    print("Statement -", testing_texts)
    
    # testing_texts = testing_texts.split(". ")
    # testing_texts = sent_tokenize(testing_texts)
    statement = testing_texts
    ##################################################################

    # texts = statement.split('. ')
    texts = sent_tokenize(statement)
    print("After sent_tokenizer -", texts)
    # input("Press ENTER key to continue...")
    
    kb = KB()
    max_lenn = 1000
    spann = 64

    for text in tqdm(texts):
        kb = ner.from_text_to_kb(text, "", kb = kb,
                            verbose=0,
                            span_length=spann,
                            max_doc_text=max_lenn,
                            useWiki=1,
                            offlineWiki=offline_wikipedia)
    kb.print()
    
    print("Starting RAG ... ")
    # input("Press ENTER key to continue...")

    user_qn = statement
    # RAG 
    rag_vector_model_name = "sentence-transformers/all-MiniLM-l6-v2"
    rag = RAG(user_qn=user_qn, kb=kb, 
              offline_wikipedia=offline_wikipedia,
              vector_model_name=rag_vector_model_name, device=device)
    rag.get_docs()
    rag.split_data_to_chunks()
    rag.store_vector()

    # input("Press ENTER key to continue...")
    print("loading LLama3 model ...")

    llama3_rag= llama3()

    print("llama3 loaded...")
    # input("Press ENTER key to continue...")

    # print(rag.get_similar())
    print("Starting Checking ...")
    start_time = time.time()
    sentence_truth = {}
    for sent_no, s in enumerate(texts):
        relevant_docs = rag.get_similar(user_qn=s)

        contextt = '\n'.join(['[DOC ' + str(i) + '] : '+ docc.page_content 
                                                for i, docc in enumerate(relevant_docs)])
        prompt = f"""<<CONTEXT>>\n{contextt}\n\n<<CHECK>> {s}"""
        print("\nprompt = \n", prompt)
        
        # LLAMA3

        llama_response = llama3_rag.llama3_summary(s, 
                                                    contextt, 
                                                    verbose=1)
        print("response :", llama_response)
        
        sentence_truth[sent_no] = llama3_rag.clean_output(llama_response)
        print(sentence_truth)
    end_time = time.time()


print(llama3_rag.calc_truth(sentence_truth))
print(f"Took {end_time - start_time} seconds")

