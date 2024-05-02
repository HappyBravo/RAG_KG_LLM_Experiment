from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class llama3():
    def __init__(self, useGPU = True):
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.device = torch.device(f"cuda:{0}" if torch.cuda.is_available() and useGPU else "cpu")
        self.model, self.tokenizer = self.model(self.model_name)

    def model(self, model_name):    
        tokenizer = AutoTokenizer.from_pretrained(model_name)    
        model = AutoModelForCausalLM.from_pretrained(
                                            model_name,
                                            # load_in_4bit = True,
                                            # low_cpu_mem_usage=True,
                                            # torch_dtype="auto",
                                            torch_dtype=torch.float16,
                                            # torch_dtype=torch.bfloat16,
                                            device_map='auto',
                                            # device_map='balanced', # When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or 'auto', 'balanced', 'balanced_low_0', 'sequential'
                                            )
        return model, tokenizer
    
    # def llama3_summary(self, comments, verbose = False):
    def llama3_eng_translator(self, statement, verbose = False):
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant who translates the text given by user from any language to English. \
                    Text given by user starts after <<TEXT>>.""" 

            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "\n" + "<<TEXT>> " + statement + "\n"
                    }
                ]
            }
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
                                messages, 
                                add_generation_prompt=True,
                                return_tensors="pt",
                                verbose=True
                        ).to(self.model.device)


        terminators = [
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]
        
        if verbose:
            print("\n --- Generating outputs --- \n")
        
        outputs = self.model.generate(
                                input_ids,
                                max_new_tokens=512,
                                eos_token_id=terminators,
                                do_sample=False,
                                temperature=0.2,
                                top_p=0.3,
                                top_k=20,
                                )
        if verbose:                    
            print("\n --- Got some response --- \n ")

        response = outputs[0][input_ids.shape[-1]:]

        if verbose:
            print("\n --- Returning after decoding response... --- \n")

        return self.tokenizer.decode(response, skip_special_tokens=True)
    
    def llama3_summary(self, statement, context = "", verbose = False):

        # text = '\n'.join([f"Author: {c['author']}\nDate: {c['datetime']}\n{c['text']}" for c in comments])
        
        # text = comments

        # messages = [
        #     {
        #         "role": "system",
        #         # "content": "You are helpful assistant that summarizes the comments section of Hacker News."
        #         "content": "You are helpful assistant who answers in one word only. You learn from <<CONTEXT>> and checks if the facts in <<CHECK>> are correct or not. If the facts are corrent then retrun 'True' otherwise return 'False'. If you cannot determine then return 'PantsOnFire'.\
        #                     Do not return anything other then 'True', 'False' or 'PantsOnFire'. "

        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "text",
        #                 "text": text
        #             }
        #         ]
        #     }
        # ]
        
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are helpful assistant who answers with some reason. Use you knowledge and also learn from <<CONTEXT>> and check if the facts in <<CHECK>> are correct or not. If the facts are corrent then return 'True' otherwise return 'False'. If you cannot determine then return 'PantsOnFire'.\
        #                     You also mention the reason for your answer.\
        #                     Do not return anything other then << 'True', 'False' or 'PantsOnFire' >> with the << reason >>." 

        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "text",
        #                 "text": text
        #             }
        #         ]
        #     }
        # ]
        # who answers who returns 'True', 'False' or 'PantsOnFire' with reason
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant who answers in 'True', 'False' or 'PantsOnFire' only.\
                You will be given context starting with '<<CONTEXT>>' and user query starting with '<<QUERY>>'\
                Check if the context and user query are not related then return 'PantsOnFire'.\
                Use your knowledge and the provided context to check the facts in user query. If the facts are correct you return 'True' else 'False'.\
                """ 

            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<<CONTEXT>> " + context + "\n\n"+"<<QUERY>> " + statement + "\n"
                    }
                ]
            }
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
                                messages, 
                                add_generation_prompt=True,
                                return_tensors="pt",
                                verbose=True
                        ).to(self.model.device)


        terminators = [
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                        ]
        
        if verbose:
            print("\n --- Generating outputs --- \n")
        
        outputs = self.model.generate(
                                input_ids,
                                max_new_tokens=64,
                                eos_token_id=terminators,
                                do_sample=True,
                                temperature=0.2,
                                top_p=0.3,
                                top_k=20,
                                )
        if verbose:                    
            print("\n --- Got some response --- \n ")

        response = outputs[0][input_ids.shape[-1]:]

        if verbose:
            print("\n --- Returning after decoding response... --- \n")

        return self.tokenizer.decode(response, skip_special_tokens=True)