import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import RobertaTokenizer

from transformers.tokenization_utils_base import BatchEncoding

class YouTubeDataset(Dataset):

    def __init__(self) -> None:
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
                if entry["engagement_rate"] <= 10:
                    self.label.append(0)
                elif 10 < entry["engagement_rate"] <= 20:
                    self.label.append(1)
                elif 20 < entry["engagement_rate"] <= 30:
                    self.label.append(2)
                else:
                    self.label.append(3)
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
