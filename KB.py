# IMPORTS 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import wikipedia
from fuzzywuzzy import fuzz
from joblib import Parallel, delayed 
import pickle, os, random
from pyvis.network import Network
import networkx as nx


N_JOB_COUNT = 1

class KB():
    def __init__(self):
        self.entities = {} # { entity_title: {...} }
        self.relations = [] # [ head: entity_title, type: ..., tail: entity_title,
          # meta: { article_url: { spans: [...] } } ]
        self.sources = {} # { article_url: {...} }

    def merge_with_kb(self, kb2):
        for r in kb2.relations:
            article_url = list(r["meta"].keys())[0]
            source_data = kb2.sources[article_url]
            self.add_relation(r, source_data["article_title"],
                              source_data["article_publish_date"])

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def merge_relations(self, r2):
        r1 = [r for r in self.relations
              if self.are_relations_equal(r2, r)][0]

        # if different article
        article_url = list(r2["meta"].keys())[0]
        if article_url not in r1["meta"]:
            r1["meta"][article_url] = r2["meta"][article_url]

        # if existing article
        
        else:
            spans_to_add = [span for span in r2["meta"][article_url]["spans"]
                            if span not in r1["meta"][article_url]["spans"]]
            r1["meta"][article_url]["spans"] += spans_to_add
            

    def get_wikipedia_data(self, candidate_entity, useWiki = True, offline_wiki = None, offline_only = False, verbose = False, redirect_count = 0):
        # print("\n\n--- offline", offline_Wiki)
        entity_data = None
        stop_words = set(stopwords.words('english'))
        if len(candidate_entity.split()) > 4:
            word_tokens = word_tokenize(candidate_entity)
            candidate_entity = " ".join([w for w in word_tokens if not w.lower() in stop_words])

        try:
            if offline_wiki:
                if verbose:
                    print(f"Finding {candidate_entity} in offline Wiki")
                _entity_data = offline_wiki.word_match(candidate_entity, verbose = verbose)
                
                if verbose:
                    print(f"Got {_entity_data} after word_match from offline Wiki")

                if "REDIRECT".lower() in _entity_data["summary"][:10].lower():
                    # entity_data = _entity_data
                    _word = _entity_data["url"].split("/wiki/")[-1].strip()
                    if verbose:
                        print(f"REDIRECT found !!! Candidate entitiy {candidate_entity} === changed to ==> {_word}")
                    if redirect_count > 5:
                        return None
                    entity_data = self.get_wikipedia_data(_word, useWiki=useWiki, offline_wiki=offline_wiki, offline_only=offline_only, verbose=verbose, redirect_count=redirect_count+1)
                else:                    
                    ratioo = fuzz.ratio(candidate_entity, _entity_data['title'])
                    if verbose:
                        print(f"Fuzz ration : {ratioo}")
                    if ratioo > 50 :
                        entity_data = _entity_data
                        if verbose:
                            print(f"Got {entity_data} from offline wiki with similarity ration = {ratioo}.")
            
            if useWiki and not entity_data and not offline_only:
                # if verbose:
                print(f"Finding {candidate_entity} in online Wiki")
                page = wikipedia.page(candidate_entity, 
                                        auto_suggest=False, 
                                        redirect=False,
                                        )
                # if page.exists():
                entity_data = {
                    "title": page.title,
                    "url": page.url,
                    "summary": page.summary
                    }
                # else:
                    # return 
                # except wikipedia.DisambiguationError as e:
                    # s = random.choice(e.options)
                    # page = wikipedia.page(s, auto_suggest= False)
                    # entity_data = {
                    #     "title": page.title,
                    #     "url": page.url,
                    #     "summary": page.summary
                    # }
                    # print("DisambiguationError")
                    # return None
                # except Exception as e:
                #     print(f"Error retriving Online Wikipedia. Error - {e}")
                #     entity_data = None
                #     return 
            return entity_data
        except:
            return None

    def add_entity(self, e):
        self.entities[e["title"]] = {k:v for k,v in e.items() if k != "title"}

    def add_relation(self, r, article_title, article_publish_date, 
                     useWiki = True, offlineWiki = None, offline_only = False, verbose = False):
        # check on wikipedia
        candidate_entities = [r["head"], r["tail"]]
        if verbose:
            print(f"Candidate entities : {candidate_entities}")
            
        # entities = [self.get_wikipedia_data(ent) for ent in candidate_entities]
        
        # TRY 2
        entities = []
        if useWiki:
            try : 
                entities = Parallel(n_jobs=N_JOB_COUNT, timeout=300)(delayed(self.get_wikipedia_data)(ent, useWiki, offlineWiki, offline_only, verbose=verbose) for ent in candidate_entities)
                # entities = [self.get_wikipedia_data(ent, useWiki, offlineWiki, verbose=verbose) for ent in candidate_entities]
            except:
                print(f"An error occurred while fetching entities data from Wiki : {e}")
                entities = None
        else:
            entities = [{"title": ent,
                         "url": "",
                         "summary": ""
                        } for ent in candidate_entities]

        # if one entity does not exist, stop
        if any(ent is None for ent in entities):
            return

        # manage new entities
        for e in entities:
            self.add_entity(e)

        # rename relation entities with their wikipedia titles
        r["head"] = entities[0]["title"]
        r["tail"] = entities[1]["title"]

        # add source if not in kb
        article_url = list(r["meta"].keys())[0]
        if article_url not in self.sources:
            self.sources[article_url] = {
                "article_title": article_title,
                "article_publish_date": article_publish_date
            }

        # manage new relation
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r)

    def print(self):
        print("Entities:")
        for e in self.entities.items():
            print(f"  {e}")
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")
        print("Sources:")
        for s in self.sources.items():
            print(f"  {s}")
    
    
    def save_network_html(self, kb, filename="network.html", 
                        verbose = False, 
                        physics = False,
                        show = False):

        if not os.path.exists(filename):
            with open(filename, 'w') as _file:
                _file.write("")

        # create network
        G = nx.Graph()
        net = Network(directed=True, 
                    notebook=True,
                    width="1000px", 
                    height="1000px",
                    #   bgcolor="#eeeeee"
                    )
        if verbose:
            print("Network initialized")

        # nodes
        color_entity = "#00FF00"
        if verbose:
            print(f"there are {len(kb.entities)} entities in KB")
        for e in kb.entities:
            G.add_node(e)
            net.add_node(e, label=e, shape="dot", color=color_entity)
            # net.add_node(e, label=e, physics = physics, shape="dot", color=color_entity)
        
        # edges
        if verbose:
            print(f"there are {len(kb.relations)} relations in KB")
        
        # for r in kb.relations:
        #     G.add_edge(r['head'], r["tail"], )
        #     # net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])
        labels = {}
        for r in kb.relations:
            G.add_edge(r['head'], r["tail"])
            labels[(r["head"], r["tail"])] = r["type"]
            net.add_edge(r["head"], r["tail"], title=r["type"], label=r["type"])

        scale=10 # Scaling the size of the nodes by 10*degree
        d = dict(G.degree)

        pos = nx.spring_layout(G)
        #Updating dict
        d.update((x, scale*y) for x, y in d.items())

        #Setting up size attribute
        nx.set_node_attributes(G,d,'size')
        nx.set_edge_attributes(G,labels, 'labels')
        # nx.draw_networkx_edge_labels(
        #                             G, pos,
        #                             edge_labels=labels,
        #                             # font_color='red'
        #                             )
        if verbose:
            print(f"Trying to make graph")

        # net.from_nx(G)   
        
        # save network
        if physics:
            net.repulsion(
                node_distance=200,
                central_gravity=0.3,
                spring_length=200,
                spring_strength=0.05,
                damping=0.09
            )

        net.set_edge_smooth('dynamic')

        if verbose:
            print(f"Trying to show graph")

        net.show(filename)

    def save_kb(self, kb, filename, verbose = False):
        if verbose:
            print(f"there are {len(kb.entities)} entities in KB")
            print(f"there are {len(kb.relations)} relations in KB")

        with open(filename, "wb") as f:
            pickle.dump(kb, f)

    def load_kb(self, filename):
        res = None
        with open(filename, "rb") as f:
            res = pickle.load(f)
        return res
