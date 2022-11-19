from transformers.optimization import get_linear_schedule_with_warmup
from transformers import T5Tokenizer,T5ForConditionalGeneration
from torch.utils.data import DataLoader
from src.dataprocessor import DataSet
from sacrebleu import raw_corpus_bleu
import pandas as pd
import torch
import os
from tqdm.auto import tqdm


class t5Trainer:

    def __init__(self,configs,logger):
        """   
        

        """
        self.configs = configs
        self.model_configs = self.configs['model_configs']
        self.data_configs = self.configs['data_configs']
        self.train_configs = self.configs['training_configs']
        self.logger = logger

        if self.model_configs['tokenizer_path']:

            self.tokenizer = T5Tokenizer.from_pretrained(self.model_configs['tokenizer_path'])
            self.logger.info("loaded tokenizer from {path} succesfully".format(path=self.model_configs['tokenizer_path']))

        else:
            self.logger.warning("tokenizer path not set, loading from default...")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_configs['model_name'])


        self.device = torch.device('cuda' if self.train_configs['use_cuda'] and torch.cuda.is_available() else 'cpu')

    
    def _load_model(self):

        if self.model_configs['model_path']:

            self.model = T5ForConditionalGeneration.from_pretrained(self.model_configs['model_path'],max_length=512)
            self.logger.info("loaded model from path {path} succesfully".format(path=self.model_configs['model_path']))

        else:
            
            self.logger.warning("model path not set, loading from default...")
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_configs['model_name'])


        self.model.to(self.device)

    def _set_dataset(self):

        train_path = self.data_configs['train_path']
        dev_path = self.data_configs['dev_path']
        
        train_df = pd.read_csv(train_path)
        dev_df = pd.read_csv(dev_path)

        train_df = DataSet(train_df,self.tokenizer,self.data_configs)
        test_df = DataSet(dev_df,self.tokenizer,self.data_configs)

        train_loader = DataLoader(train_df,batch_size=self.train_configs['batch_size'],shuffle=True)
        dev_loader = DataLoader(test_df,batch_size = self.train_configs['batch_size']//2,shuffle=False)

        return train_loader,dev_loader

    def _set_optimizer(self,model_param):

        if self.train_configs['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model_param,self.train_configs['lr'],weight_decay = self.train_configs['weight_decay'],betas=tuple(self.train_configs['betas']),eps=self.train_configs['eps'])
            self.logger.info("adam optimizer loaded succesfully")

        elif self.train_configs['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(model_param,self.train_configs['lr'],eps=self.train_configs['eps'],weight_decay=self.train_configs['weight_decay'],betas=tuple(self.train_configs['betas']))
            self.logger.info("adamW optimizer loaded successfully")

        elif self.train_configs['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(model_param,self.train_configs['lr'],weight_decay=self.train_configs['weight_decay'])
            self.logger.info("sgd optimizer loaded succesfully")

        else:
            self.logger.warning("implementation suppors only Adam, AdamW and sgd, setting optimizer to Adam")
            optimizer = torch.optim.Adam(model_param,self.train_configs['lr'],weight_decay = self.train_configs['weight_decay'],betas=tuple(self.train_configs['betas']),eps=self.train_configs['eps'])

            

        return optimizer

    def train(self):
        """
        
        
        """
        try:
            self.logger.info("loading model...")
            self._load_model()
            
        except Exception as e:
            raise e

        try:
            self.logger.info("setting optimizer...")
            optimizer = self._set_optimizer(self.model.parameters())
            
        except Exception as e:
            raise e
            

        train_loader, eval_loader = self._set_dataset()
        self.logger.info("datasets loaded to DataLoader")

        
        total_steps = int(len(train_loader.dataset) / self.train_configs['batch_size'])
        warmup_steps = int(0.1 * total_steps)

        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps,
                num_training_steps=total_steps - warmup_steps
            ) 

        best_bleu = 0.0
        self.model.train()

        for epoch in range(self.train_configs['epochs']):
            train_loss = 0
            for idx, (batch,_) in enumerate(train_loader):

                batch = {k:torch.tensor(v).to(self.device) for k,v in batch.items()}

                optimizer.zero_grad()
                loss = self.model(**batch).loss
                loss.backward()

                optimizer.step()
                scheduler.step()

                train_loss+=loss.item()

                if idx+1 % 100 == 0:
                    tqdm.write(f"Epoch {epoch + 1}/{self.train_configs['epochs']}; Batch {idx}; Loss {loss.detach().cpu().numpy()}")
                    
                
            if self.train_configs['evaluate_dev']:
                self.logger.info("evaluating dev set...")
                bleu_score = self._evaluate(eval_loader)

                if bleu_score>best_bleu:
                    self.logger.info("best bleu score, saving model to path {path} at bleu score of {bleu_score}...".formate(path=self.train_configs['path_to_save']))
                    self._save_model(self.model)
                    self.logger.info("saved to path {path}".format(path=self.train_configs['path_to_save']))

                else:
                    self.logger.info("new bleu score {bleu_score} not better than {best_bleu}, model will not be saved")
                
                self.model.train()

        self.logger.info("Training completed, model outputs saved in path {path}, at best bleu score of {best_bleu}".format(path=self.train_configs['path_to_save']))
    
    
    def _evaluate(self,loader):
        self.model.eval()
        hyps = []
        true_pred = []
        
        with torch.no_grad():

            for (batch,tgt) in loader:

                input_id = batch['input_ids'].to(self.device)
                attn_mask = batch['attention_mask'].to(self.device)

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


    
    def _save_model(self,checkpoints):
        self.logger.info("saving model...")

        save_to = os.path.join(self.train_configs['path_to_save'])
        self.logger.info("model save to path {path}".format(path=self.train_configs['path_to_save']))
        torch.save(checkpoints,save_to)
