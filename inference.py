from transformers import T5Tokenizer,T5ForConditionalGeneration
import torch


class t5inference:
    def __init__(self,configs,logger):
        self.configs = configs
        self.logger = logger
        self.model_configs = configs['model_configs']
        

        if not self.model_configs['model_path']:
            raise Exception(logger.error("could not find path to load model,please provide path to model"))
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_configs['model_path'])

        if not self.model_configs['tokenizer_path']:
            raise Exception(logger.error("tokenizer path not set, provide path to tokenizer"))
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_configs['tokenizer_path'])


        self.device = torch.device("cuda" if self.configs['training_configs']['use_cuda'] and torch.cuda.is_available() else 'cpu')


    def process_data_single(self,text):

        self.model.to(self.device)
        self.model.eval()
        prefix = self.configs['data_configs']['prefix']
        text = prefix + text

        max_len = self.configs['data_configs']['max_input_len']
        tokenized_text = self.tokenizer(text,padding='max_length',max_length=max_len,truncation=True,add_special_tokens=True,return_tensors='pt')

        input_id = tokenized_text['input_ids'].to(self.device)
        attn_mask = tokenized_text['attention_mask'].to(self.device)

        output = self.model.generate(input_ids=input_id,attention_mask=attn_mask)[0]

        decoded_text = self.tokenizer.decode(output,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        
        return decoded_text

    def process_data_batch(self,texts):

        self.model.to(self.device)
        self.model.eval()
        prefix = self.configs['data_configs']['prefix']

        texts = [prefix+text for text in texts]

        max_len = self.configs['data_configs']['max_input_len']
        tokenized_text = [self.tokenizer(translation,padding='max_length',max_length=max_len,truncation=True,add_special_tokens=True,return_tensors='pt') for translation in texts]
        
        outputs = [self.model.generate(input_ids = tokens['input_ids'].to(self.device),attention_mask=tokens['attention_mask'].to(self.device)) for tokens in tokenized_text]
        
        decoded_texts = [self.tokenizer.decode(output[0],skip_special_tokens=True,clean_up_tokenization_spaces=False) for output in outputs]

        return decoded_texts


    def infer_batch(self,texts):
        output_text = self.process_data_batch(texts)
        return output_text

    def infer_single(self,texts):

        output_text = self.process_data_single(texts)

        return output_text