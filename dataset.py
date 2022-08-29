import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import RobertaTokenizer

from transformers.tokenization_utils_base import BatchEncoding

class YouTubeDataset(Dataset):

    def __init__(self, splits) -> None:
        super().__init__()
        self.id2idx = {}
        self.text = []
        self.label = []
        self.id = []
        self.tokens = None
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        captionDir = "Data/CaptionPunctuated/"

        df = pd.read_csv("Data/youtubeDataset.csv")
        cnt = 0
        for _, entry in df.iterrows():
            # dataEntry = {
            #     "id": entry["RECORD ID"],
            #     "label": int(entry["engagement_rate"])
            # }
            if entry["engagement_rate"] != 0:
                self.id.append(entry["RECORD ID"])
                if entry["engagement_rate"] > splits[-1]:
                  self.label.append(len(splits))
                elif entry["engagement_rate"] < splits[0]:
                  self.label.append(0)
                else:
                  for i in range(len(splits)-1):
                    if entry["engagement_rate"] > splits[i] and entry["engagement_rate"] < splits[i-1]:
                      self.label.append(i+1)
                      break
                with open(captionDir+entry["RECORD ID"]+".txt",'r') as f:
                    self.text.append(f.read())
            self.id2idx[entry["RECORD ID"]] = cnt
            cnt += 1
        self.label = torch.tensor(self.label,dtype=torch.int64)
        self.tokens = self.tokenizer(self.text, max_length = 512, padding=True, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index,int):
            index = [index]
        return BatchEncoding(
            {
                'input_ids': self.tokens['input_ids'][index],
                'attention_mask': self.tokens['attention_mask'][index]
            }
        ), self.label[index]
