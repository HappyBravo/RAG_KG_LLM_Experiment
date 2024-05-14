# IMPORT
import os, json, pickle, re, math, torch, multiprocessing, nltk, time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd

import wikipedia
from joblib import Parallel, delayed 
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

from google_search_util import GoogleUtil

from llama3 import llama3

from checkInternetConn import is_connected

from mylogger import Logger

log_folder = "./Logs/"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
    print(f"{log_folder} - log folder created...")

results_folder = "./results/"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    print(f"{log_folder} - results folder created...")

time_during_start = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = log_folder + f"log_{time_during_start}.txt"
logger = Logger(log_file=log_filename)


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
# N_JOB_COUNT = 4
logger.log(f"{N_JOB_COUNT} CPU threads will be used for some processes.")

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
logger.log(f"Using device : {device}")

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

def loadCSV(filepathh):
    if os.path.exists(filepathh):
        df = pd.read_csv(filepathh)
        return df
    else:
        print(f"{filepathh} does not exist...\n")
        return None
    
def loadFILE(filepathh = ""):
    if os.path.exists(filepathh):
        if filepathh.endswith(".txt"):
            return loadTXT(filepathh)
        elif filepathh.endswith(".json"):
            return loadJSON(filepathh)
        elif filepathh.endswith(".csv"):
            return loadCSV(filepathh)
        else:
            print("\n- Invalid File format 😐 !!!\n")
            return None
    else:
        print(f"{filepathh} does not exists...\n") 


if __name__ == "__main__":

    # LOADING WIKI
    WIKI_INDEX_FILE = "D://WikiDump/enwiki-20240220-pages-articles-multistream-index.txt/enwiki-20240220-pages-articles-multistream-index.txt"
    WIKI_BZ2_FILE = "D://WikiDump/enwiki-20240220-pages-articles-multistream.xml.bz2"

    INDEX_FOLDER = "./indexes/"

    assert all(map(os.path.exists, [WIKI_INDEX_FILE, WIKI_BZ2_FILE, INDEX_FOLDER]))

    offline_wikipedia = offline_Wiki(wiki_index_file=WIKI_INDEX_FILE,
                                    wikiDump_bz2_file=WIKI_BZ2_FILE, 
                                    index_folder=INDEX_FOLDER,
                                    verbose=False,)
    
    logger.log("Offline Wikipedia loaded.")
    
    print("Loading NER model ... ")
    logger.log("Loading NER model ... ")

    # Load model and tokenizer for NER
    ner_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    ner_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
    ner_model.to(device)

    ner = NER(model=ner_model, tokenizer=ner_tokenizer, device = device)
    print("NER module initialized ... ")
    logger.log("NER module initialized ... ")

    # TESING KB
    
    # input("Press ENTER key to continue...")
    
    # print("Testing KB and NER ... ")

    # statement = "The Sun rises in East direction. The Earth is smaller than the planet Jupiter. Tigers are from Cat family."
    # print("Statement -", statement)

    use_Google = 1
    logger.log(f"use_Google = {use_Google}")

    # RAG 
    print("Loading RAG model ... ")
    logger.log("Loading RAG model ... ")

    rag_vector_model_name = "sentence-transformers/all-MiniLM-l6-v2"
    
    rag = RAG(offline_wikipedia=offline_wikipedia,
              vector_model_name=rag_vector_model_name, 
              device=device)

    print("RAG module initialized ... ")
    logger.log("RAG module initialized ... ")

    print("Loading LLama3 model ...")
    logger.log("Loading LLama3 model ...")

    llama3_rag = llama3()

    print("LLama3 loaded...")
    logger.log("LLama3 loaded...")

    # input("Press ENTER key to continue...")


    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # testing_file = "./data/averitec_claims.csv"
    # testing_file = "./data/politifact_claims.csv"
    testing_file = "./data/snopes_claims.csv"
    
    assert os.path.exists(testing_file)

    testing_df = pd.read_csv(testing_file)

    # _input_statement = input("Enter the news to check : ").strip()
#     _input_statement = ["""
# New report claims India's rise on world stage under PM Modi is 'a mirage'. It claims that in both the US and UK, Modi is neither well known nor popular, and refers to a recent YouGov poll in
# which he was ranked in both countries below the highly disliked figures in those countries of Vladimir Putin and Xi Jinping. In UK just 10% view Modi favourably, the poll, based on a sample of adults, claims.
# It cites another poll from YouGov that found 80% of Indians are concerned for the health of their democracy. It blames a lack of press freedom in India as the reason for the mirage and calls on India to change course toward greater respect for human rights and democratic norms.
# The report claims that 52% of British Indians don't like Modi, based on polling by GQR, and that 65% of British Indians rate religious violence allegedly promoted by Modi spilling over to the UK as a top concern.
# The report also claims the majority of people in the US, UK and France are concerned about the state of human rights and democracy in India, alleged attempts by India to assassinate US and Canadian citizens on home soil, and new laws which make it harder for Muslims to become citizens of India, and says they want to see human rights as conditions of trade deals.

# "State machinery is being used to oppress the opposition...while BJP leaders enjoy total impunity," it claims.
# # """.replace('\n', ' ')]
#     _label = [""]
#     _testing_df = {"Claim" : _input_statement,
#                   "Label_mapped" : _label}
    
#     testing_df = pd.DataFrame(_testing_df)

    claim_label_tuples = [(claim, label) for claim, label in zip(testing_df["Claim"], testing_df["Label_mapped"])]
    
    logger.log(f"'{testing_file}' loaded...")
    
    # n_claims_start = 13
    # n_claims_end = 20

    n_claims_start = int(input("Enter starting : "))
    n_claims_end = int(input(f"Enter ending (after {n_claims_start + 1} and under {len(claim_label_tuples)}) : "))

    resulting_file = results_folder + f"result_{time_during_start}.csv"
    with open(resulting_file, "a") as result_f:
        result_f.write("Claim,Label_mapped,Result\n")
    _result = {}
    
    logger.log(f"Running for {n_claims_start} - {n_claims_end} claims.")
    countt = 0
    for claim, label in claim_label_tuples[n_claims_start : n_claims_end]:
        statement = claim

        print("Claim -", statement)
        logger.log(f"Claim - {statement}")

        # input("Press ENTER key to continue...")

        # texts = statement.split('. ')
        texts = sent_tokenize(statement)
        lengths = [len(_sentence.split()) for _sentence in texts]
        avg_len = sum(lengths)/len(texts)
        if avg_len < 5 or max(lengths) > 10:
            splitter = RecursiveCharacterTextSplitter(chunk_size = 64, chunk_overlap = 24)
            texts += splitter.split_text(statement)
        print("After sent_tokenizer -", texts)
        logger.log(f"After sent_tokenizer - {texts}")

        # input("Press ENTER key to continue...")
        
        kb = KB()
        max_lenn = 1000
        spann = 128

        # for text in tqdm(texts):
        for text in texts:
            print(f"Doing NER of - '{text}'")
            logger.log(f"Doing NER of - '{text}'")
            text = text.replace("-", " ")   # "-" IS CAUSING ISSUE IN NER ... 
                                            # EXAMPLE: IN "COVID-19"
            
            kb = ner.from_text_to_kb(text, "", kb = kb,
                                verbose=0,
                                span_length=spann,
                                max_doc_text=max_lenn,
                                useWiki=1,
                                # offline_only = 1,
                                offlineWiki=offline_wikipedia)
        kb.print()
        
        print("Starting RAG process... ")
        logger.log("Starting RAG process... ")

        # input("Press ENTER key to continue...")

        user_qn = statement

        if not kb.entities:
            print("No entities found by NER process ... \nChecking next Claim...\n\n")
            logger.log("No entities found by NER process ... \nChecking next Claim...\n\n")
            continue

        _docs, _data = rag.get_wiki_docs(entities=kb.entities)
        _chunks = rag.split_data_to_chunks(data = _data)
        _vector_store = rag.store_vector(chunks=_chunks)

        logger.log("Offline Wiki docs vectorised ...")

        is_connected_to_net = is_connected(time_out = 60)
        print(f"Connected to Net - {is_connected_to_net}")
        logger.log(f"Connected to Net - {is_connected_to_net}")
        
        if is_connected_to_net and use_Google:
            googleBaba = GoogleUtil()

            logger.log("Google Utility started...")
            
            search_dict = {}
            news_dict = {}

            relations = [sorted((relation['head'], relation['tail'])) for relation in kb.relations if ((relation['head'] != relation['tail']) 
                                                                                                    or ('time' in relation['type'].lower()))]
            relations.sort()
            relations = list(set(map(tuple, relations)))

            print(relations)
            logger.log(f"Relations : {relations}")

            # input("Press ENTER key to continue...")
            fetched_data = []

            # search_dict, news_dict = googleBaba.fetch_data_from_relation(relations=relations)
            
            n = max(math.ceil(len(relations)*0.25), 2)
            
            try :
                fetched_data = Parallel(n_jobs=N_JOB_COUNT, timeout=600)(
                                        delayed(
                                            googleBaba.fetch_data_from_relation)(relations=_relations) 
                                                                    for _relations in [relations[i:i+n] 
                                                                                        for i in range(0, len(relations), n)])
            except Exception as e:
                print(f"An error occurred while fetching data from Google : {e}")
                logger.log(f"An error occurred while fetching data from Google : {e}")
                fetched_data = []

            # print(fetched_data)
            # input("Press ENTER key to continue...")
            print("Fetched required data from Google...")
            logger.log("Fetched required data from Google...")
            for processed in fetched_data:
                # 0th is search_data
                # 1th is news_data
                search_dict.update(processed[0])
                news_dict.update(processed[1])    
            
            search_data = "\n\n".join([value for value in search_dict.values()])
            news_data = "\n\n".join([value for value in news_dict.values()])

            # print(news_data)
            # input("Press ENTER key to continue...")

            # print(search_data)
            # input("Press ENTER key to continue...")

            big_data = news_data + search_data
            googleBaba_chunks = rag.split_data_to_chunks(big_data)

            googleBaba_vector = rag.store_vector(chunks=googleBaba_chunks)
    
        # input("Press ENTER key to continue...")
        
        # print(rag.get_similar())
        print("Starting Checking ...")
        logger.log("Starting Checking ...")
        start_time = time.time()
        sentence_truth = {}
        for sent_no, s in enumerate(texts):
            relevant_docs = []

            trying_with_google = False 
            while True:
                if not relevant_docs:
                    relevant_docs = rag.get_similar(user_qn=s, 
                                                    vector_store=_vector_store)
                
                googleBaba_relevant_docs = []
                if trying_with_google and (use_Google and is_connected_to_net):
                    googleBaba_relevant_docs = rag.get_similar(user_qn=s, 
                                                               vector_store=googleBaba_vector, k = 5)
                    relevant_docs = googleBaba_relevant_docs + relevant_docs[:3]

                contextt = '\n'.join(['[DOC ' + str(i) + '] : '+ docc.page_content 
                                                        for i, docc in enumerate(relevant_docs)])
                prompt = f"""<<CONTEXT>>\n{contextt}\n\n<<CHECK>> {s}"""
                print("\nprompt = \n", prompt)
                logger.log(f"\nprompt = \n{prompt}")
                
                # LLAMA3
                llama_response = llama3_rag.llama3_summary(s, 
                                                            contextt, 
                                                            verbose=1)
                print("response :", llama_response)
                logger.log(f"Response : {llama_response}")
                
                sentence_truth[sent_no] = llama3_rag.clean_output(llama_response)
                print(sentence_truth)
                logger.log(f"{sentence_truth}")

                if trying_with_google or not is_connected_to_net:
                    break

                if sentence_truth[sent_no] not in [0, 1]:
                    # CHECK GOOGLE
                    if is_connected_to_net and use_Google:
                        print("Lets try with Google Search and Google News data...")
                        logger.log("Lets try with Google Search and Google News data...")
                    trying_with_google = True
                    continue
                break
                
        end_time = time.time()

        _result = llama3_rag.calc_truth(sentence_truth)
        print(_result)
        logger.log(f"{_result}")

        with open(resulting_file, "a") as result_f:
            result_f.write(f"{claim},{label},{_result}\n")

        print(f"Took {end_time - start_time} seconds\n\n")
        logger.log(f"Took {end_time - start_time} seconds\n\n")

        countt += 1
    
    print(f"Checked {countt} Claims...")
    logger.log(f"Checked {countt} Claims...")
    print("Exiting ...")
    input("ENTER ANY KEY TO CONTINUE...")
    logger.log("Exiting ...")



