from transformers import T5Tokenizer,T5ForConditionalGeneration
from src.dataprocessor import DataSet
import pandas as pd
from torch.utils.data import DataLoader
from sacrebleu import raw_corpus_bleu
import torch



class T5Evaluator:
    def __init__(self,configs,logger):

        self.data_configs = configs['data_configs']
        self.model_configs = configs['model_configs']
        self.logger = logger
        

        if self.model_configs['tokenizer_path']:
            self.logger.info("loading tokenizer...")

            self.tokenizer = T5Tokenizer.from_pretrained(self.model_configs['tokenizer_path'])
        else:
            raise Exception(self.logger.error("tokenizer path not set, please provide path to tokenizer in config file"))

        if self.model_configs['model_path']:
            self.logger.info("loading model...")
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_configs['model_path'])

        else:
            raise Exception(self.logger.error("model path not set, please provide path to model in config file"))

        

        self.device = torch.device('cuda' if configs.training_configs['use_cuda'] and torch.cuda.is_available() else 'cpu')


    def evaluate(self,batch_size):
        if not self.data_configs['dev_path']:
            raise Exception(self.logger("please provide path to dev path for evaluation"))

        else:
            df = pd.read_csv(self.data_configs['dev_path'])
           
            eval_ds = DataSet(df,self.tokenizer,self.data_configs)
            eval_loader = DataLoader(eval_ds,batch_size=batch_size,shuffle=True)

            
            bleu_score = self._evaluate(eval_loader)
            self.logger.info("eval set bleu score: ",bleu_score)

        
            


    def _evaluate(self,loader):
        self.model.eval()
        self.model.to(self.device)
        hyps = []
        true_pred = []
        
        with torch.no_grad():

            for (batch,tgt) in loader:

                input_id = batch['input_ids'].to(self.device)
                attn_mask = batch['attention_mask'].to(self.device)

                print("input_ids: ",input_id.shape)
                print("attn_mask: ",attn_mask.shape)

                output = self.model.generate(input_ids=input_id,attention_mask=attn_mask,max_length=200)

                decoded = [self.tokenizer.decode(translation,skip_special_tokens=True,clean_up_tokenization_spaces=False) for translation in output]

                hyps+=decoded
                true_pred+=tgt

                bleu_score = self._calculate_bleu_score(hyps,true_pred)
                

        return bleu_score


    def _calculate_bleu_score(self,hyps: list, refs: list) -> float:
        """
        calculates bleu score.

        """
        assert len(refs) == len(hyps), "no of hypothesis and references sentences must be same length"
        bleu = raw_corpus_bleu(hyps, [refs])
        return bleu.score