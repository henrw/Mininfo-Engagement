from torch import nn
import torch
from transformers import RobertaModel, RobertaTokenizer, BertTokenizer, BertModel
from .base import *

class RobertaClassifier(nn.Module):

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reset()
        test_message = "Initialization success."
        print(f"Initialization success if you see a tensor: {self.forward(test_message)}.")
    
    def reset(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.base = RobertaModel.from_pretrained('roberta-base')
        self.linear1 = nn.Linear(768,10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10,self.num_classes)

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

class BertClassifier(nn.Module):

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reset()
        test_message = "Initialization success."
        print(f"Initialization success if you see a tensor: {self.forward(test_message)}.")
    
    def reset(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.base = BertModel.from_pretrained("bert-base-uncased")
        self.linear1 = nn.Linear(768,10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10,self.num_classes)

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

class Wav2VecClassifier(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reset()
        # test_message = "Initialization success."
        # print(f"Initialization success if you see a tensor: {self.forward('data/audio/MI0001.wav')}.")
    
    def reset(self):
        self.base = Wav2Vec()
        self.linear1 = nn.Linear(768,10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10,self.num_classes)

    def forward(self, input):
        if isinstance(input, str):
            waveform, sample_rate = torchaudio.load(input)
            if sample_rate != self.base.bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.base.bundle.sample_rate)
        else:
            waveform = input
        output = self.base(waveform).mean(dim=-2)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)

        return output
    
    def predict(self, token):
      digits = self.forward(token)
      return digits.argmax()