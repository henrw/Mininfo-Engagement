from torch import nn
import torch
from transformers import RobertaModel, RobertaTokenizer
from .base import *

class RobertaWav2VecClassifier(nn.Module):

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.base_bert = RobertaModel.from_pretrained('roberta-base')
        self.base_wav2vec = Wav2Vec()
        self.linear1 = nn.Linear(768*2,10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10,self.num_classes)

    def forward(self, token, audio):
        # print("1",audio.shape)
        # if isinstance(token,str):
        #     token = self.tokenizer(token, return_tensors="pt")
        # elif isinstance(token, list):
        #     token = self.tokenizer(token, return_tensors="pt", max_length = 512, padding=True, truncation=True)
        
        # if isinstance(audio, str):
        #     waveform, sample_rate = torchaudio.load(audio)
        #     if sample_rate != self.base_wav2vec.bundle.sample_rate:
        #         waveform = torchaudio.functional.resample(waveform, sample_rate, self.base_wav2vec.bundle.sample_rate)
        # else:
        #     waveform = audio

        output_bert = self.base_bert(**token)
        # print("2",waveform.shape)
        output_wav2vec = self.base_wav2vec(audio).mean(dim=-2).to('cuda:3')
        # print(output_bert['pooler_output'].shape)
        # print(output_wav2vec.shape)
        # if output_wav2vec.shape[0] != 5:
        #     return None
        output = torch.cat([output_bert['pooler_output'], output_wav2vec],dim=1)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)

        return output
    
    def predict(self, token):
      digits = self.forward(token)
      return digits.argmax()