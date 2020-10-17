import torch
from transformers.modeling_t5 import T5ForConditionalGeneration
from transformers.tokenization_t5 import T5Tokenizer


class Summary_Predict():
    def __init__(self, model_path:str, vocab_file:str):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        self.tokenizer = T5Tokenizer(vocab_file)
    
    def predict(self, text:str):
        text = "[summary]" + text
        text = text.replace('\n', ' ')
        text = text.replace('  ', ' ')

        input_ids = torch.tensor(self.tokenizer.encode(text)).long().unsqueeze(0)
        attention_mask = input_ids.ne(0).float()
        pred = torch.tensor([0]).long().unsqueeze(0)
        enc_output = self.model.encoder.forward(input_ids=input_ids, attention_mask=attention_mask)

        eos_index = 3

        while 1:
            seq_output = self.model.decoder.forward(input_ids=pred, encoder_hidden_states=enc_output[0])[0]
            out = self.model.lm_head(seq_output)
            pred = torch.cat([pred, out.argmax(-1)[:,-1].unsqueeze(0)], dim=-1)
            if pred[0][-1].item() == 3:
                break
        
        output = self.tokenizer.decode(pred[0], skip_special_tokens=True)
        return output