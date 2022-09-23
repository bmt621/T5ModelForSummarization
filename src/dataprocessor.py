from torch import tensor 


class DataSet:
    def __init__(self,df,tokenizer,configs):

        self.src_text = df['text'].to_list()
        self.tgt_text = df['summary'].to_list()
        self.src_max_len = configs['max_input_len'] 
        self.tgt_max_len = configs['max_output_len']
        self.tokenizer = tokenizer
        self.prefix = configs['prefix']

    def __getitem__(self,idx):
        
        text = self.prefix + self.src_text[idx]
        input_tokens = self.tokenizer(text,padding='max_length',max_length=self.src_max_len,truncation=True,add_special_tokens=True)
        output_tokens = self.tokenizer(self.tgt_text[idx],padding='max_length',max_length=self.tgt_max_len,truncation=True,add_special_tokens=True).input_ids
        output_tokens = tensor(output_tokens)

        output_tokens[output_tokens==self.tokenizer.pad_token_id] =-100

        input_id = input_tokens['input_ids']
        attn_mask = input_tokens['attention_mask']
        
        return{
            'input_ids':tensor(input_id),
            'attention_mask':tensor(attn_mask),
            'labels':tensor(output_tokens).flatten()
        },self.tgt_text[idx]

    def __len__(self):
        return len(self.src_text)