from KB import KB 
import torch, math
from nltk.tokenize import word_tokenize, sent_tokenize
from joblib import Parallel, delayed 

N_JOB_COUNT = 1

class NER():
    def __init__(self, model, tokenizer, device):
        self.model = model 
        self.tokenizer = tokenizer 
        self.device = device
        # pass 

    def extract_relations_from_model_output(self, text):
        relations = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
        for token in text_replaced.split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            relations.append({
                'head': subject.strip(),
                'type': relation.strip(),
                'tail': object_.strip()
            })
        return relations

    def split_docs(self, text, 
                    max_text_count = 1000,
                    verbose = False):
        
        sentences = sent_tokenize(text)
        chunks = []
        chunk = ""
        len_sentence = 0
        chunk_len = 0
        for sentence in sentences:
            len_sentence = len(sentence.strip().split())
            if chunk_len+len_sentence < max_text_count:
                chunk += sentence+" "
                chunk_len += len_sentence
                continue
            chunk_len = len_sentence
            chunks.append(chunk)
            chunk = sentence
        
        # chunk = textwrap.wrap(sentence, max_text_count)
        if verbose:
            print(len(chunks))
        return chunks

    def _from_text_to_kb(self, text, article_url,
                        kb = None,
                        useGPU=0, 
                        span_length=128, 
                        article_title=None,
                        article_publish_date=None, 
                        verbose=False,
                        useWiki=True,
                        offline_Wiki = None):
        
        # tokenize whole text
        # print(text)
        # input()
        with torch.no_grad():
            inputs = self.tokenizer([text], 
                            max_length = 1000,
                            #    max_length=512,
                            padding=True,  
                            truncation=True, 
                            return_tensors="pt")

            # compute span boundaries
            # print(inputs.values())
            num_tokens = len(inputs["input_ids"][0])
            if verbose:
                print(f"Input has {num_tokens} tokens")
            num_spans = math.ceil(num_tokens / span_length)
            
            if verbose:
                print(f"Input has {num_spans} spans")
            overlap = math.ceil((num_spans * span_length - num_tokens) / 
                                max(num_spans - 1, 1))
            
            # input()
            spans_boundaries = []
            start = 0
            for i in range(num_spans):
                spans_boundaries.append([start + span_length * i,
                                        start + span_length * (i + 1)])
                start -= overlap
            if verbose:
                print(f"Span boundaries are {spans_boundaries}")

            # transform input with spans
            tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                        for boundary in spans_boundaries]
            tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                            for boundary in spans_boundaries]
            inputs = {
                "input_ids": torch.stack(tensor_ids),
                "attention_mask": torch.stack(tensor_masks)
            }
            
            # generate relations
            num_return_sequences = 3
            gen_kwargs = {
                "max_length": 256,
                "length_penalty": 0,
                "num_beams": 3,
                "num_return_sequences": num_return_sequences
            }

            generated_tokens = self.model.generate(
                                                inputs["input_ids"].to(self.model.device),
                                                attention_mask=inputs["attention_mask"].to(self.model.device),
                                                **gen_kwargs,
                                                )

            # decode relations
            decoded_preds = self.tokenizer.batch_decode(generated_tokens,
                                                skip_special_tokens=False)

            # create kb
            if not kb:
                kb = KB()

            i = 0
            # for sentence_pred in tqdm(decoded_preds, leave=False):
            _relations = Parallel(n_jobs=N_JOB_COUNT)(delayed(self.extract_relations_from_model_output)(sentence_pred) for sentence_pred in decoded_preds)

            for sentence_pred in decoded_preds:
                current_span_index = i // num_return_sequences
                # relations = extract_relations_from_model_output(sentence_pred)
                relations = _relations[i]

                if verbose:
                    print(f"{i}. extraction of relations done, it has {len(relations)} relations", end="\r")
                    
                for relation in relations:
                    relation["meta"] = {
                        article_url: {
                            "spans": [spans_boundaries[current_span_index]]
                        }
                    }
                    kb.add_relation(relation, 
                                    article_title,
                                    article_publish_date, 
                                    useWiki=useWiki,
                                    offlineWiki=offline_Wiki,
                                    verbose=verbose)
                i += 1

        return kb

    def from_text_to_kb(self, text, article_url,
                        kb = None,
                        useGPU=0, 
                        span_length=128, 
                        article_title=None,
                        article_publish_date=None, 
                        verbose=False,
                        max_token = 1000,
                        max_doc_text = 1000,
                        useWiki = True,
                        offlineWiki = None):
        # with torch.no_grad():
        #     # tokenize whole text
        #     # inputs = tokenizer([text], return_tensors="pt")
        #     # num_tokens = len(inputs["input_ids"][0])

        input_words = text.split()
        num_tokens = len(input_words)

        if verbose:
            # print(f"Input has {num_tokens} tokens")
            print(f"Input has {num_tokens} words")

        if not kb:
            kb = KB()
        
        _kb = kb 

        _offlineWiki = offlineWiki

        # compute span boundaries
        # num_tokens = len(inputs["input_ids"][0])
        if num_tokens > max_token:
            if verbose:
                print("input len > token size, splitting doc in smaller chunks")
            text = self.split_docs(text, max_text_count=max_doc_text)
        
        if type(text) == str:
            text = [text]
        
        # for _text in tqdm(text, leave=False):
        for _text in text: 
            # print(_text)
            # print(_text[0])
            # input()
            _kb = self._from_text_to_kb(_text, article_url, 
                                    useGPU=useGPU, 
                                    span_length=span_length, 
                                    article_title=article_title,
                                    article_publish_date=article_publish_date, 
                                    verbose=verbose,
                                    kb=_kb,
                                    useWiki=useWiki,
                                    offline_Wiki=_offlineWiki)
        return _kb


