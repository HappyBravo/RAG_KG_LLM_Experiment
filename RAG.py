# IMPORTS
from KB import KB
from wikiDumpSearch import offline_Wiki
import wikipedia

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class RAG():
    def __init__(self, user_qn, kb = None, offline_wikipedia = None, useOnlineWiki = False, 
                 chunk_size = 300, chunk_overlap = 50, 
                 vector_model_name = None, device = None,
                 verbose = False):
        
        self.user_qn = user_qn
        self.kb = kb 
        self.offline_wikipedia = offline_wikipedia
        self.verbose = verbose

        self.useOnlineWiki = useOnlineWiki
        self.docs = {} 
        self.data = ""

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks = []

        if not vector_model_name :
            vector_model_name = "sentence-transformers/all-MiniLM-l6-v2"
        self.vector_model_name = vector_model_name

        if not device :
            device = "cpu"
        self.device = device

        self.vector_store = None

        # pass 

    def get_docs(self, useOnlineWiki = False, verbose = False):
        entities_in_kb = self.kb.entities
        doc_titles = []
        docs = {}
        for _entt, _values in entities_in_kb.items():
            _url = _values['url']
            _url_word = _url.split("wiki/")[-1].strip().replace("_", " ")
            print(_url_word)
            # doc_titles.append(_url_word)
            if useOnlineWiki:  
                try:
                    docs[_url_word] = wikipedia.page(_url_word, auto_suggest=False).content.replace("\n\n", " ") # <--- NEED TO FIX THIS
                except wikipedia.exceptions.DisambiguationError :
                    pass
                continue

            # print(offline_wikipedia.word_match(_url_word, verbose=0, summaryOnly=False))
            docs[_url_word] = self.offline_wikipedia.word_match(_url_word, verbose=0, summaryOnly=False)
            if docs[_url_word]:
                docs[_url_word] = docs[_url_word]["summary"]
            else: 
                del docs[_url_word] # REMOVING THOSE ENTRIES WHICH CONTAIN None
                print(f"---{_url_word} removed")
            # --- NEED TO FIX THIS ---
        self.docs = docs 
        self.data = "".join([f"{_content}\n\n" for _word, _content in docs.items()])
        
        if verbose or self.verbose:
            print(docs)
        return docs
    
    def split_data_to_chunks(self, data = None):
        if not data:
            data = self.data
        chunks = self.text_splitter.split_text(data)
        self.chunks = chunks
        return chunks
    
    def store_vector(self, chunks = None):
        # MAKING TEXT EMBEDDING (VECTORIZING CHUNKS)
        # modelPath = "sentence-transformers/all-MiniLM-l6-v2"
        # model_kwargs = {'device': 'cpu'}
        model_kwargs = {'device': self.device}

        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
                                            model_name=self.vector_model_name, 
                                            model_kwargs=model_kwargs, 
                                            encode_kwargs=encode_kwargs 
                                            )
        if not chunks : 
            chunks = self.chunks
        # print(chunks)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        self.vector_store = vector_store
        return vector_store
    
    def get_similar(self, user_qn = "", k = 5):
        if not user_qn :
            user_qn = self.user_qn
        return self.vector_store.similarity_search(user_qn, k = k)
        