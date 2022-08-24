from torch import nn
from transformers import RobertaModel, RobertaTokenizer

class SimpleBert(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.base = RobertaModel.from_pretrained('roberta-base')
        self.linear1 = nn.Linear(768,10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10,4)
        
        test_message = "Initialization success."
        print(f"Initialization success if you see a tensor: {self.forward(test_message)}.")


    def forward(self, token):
        if isinstance(token,str):
            token = self.tokenizer(token, return_tensors="pt")
        elif isinstance(token, list):
            token = self.tokenizer(token, return_tensors="pt", max_length = 512, padding=True, truncation=True)
        output = self.base(**token)
        output = self.linear1(output['pooler_output'])
        output = self.relu(output)
        output = self.linear2(output)

        return output
    
    def predict(self, token):
      digits = self.forward(token)
      return digits.argmax()
