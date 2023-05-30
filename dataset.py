import torch
# import torchaudio
from torch.utils.data import Dataset
import pandas as pd
from transformers import RobertaTokenizer
import time
import torch.nn.functional as F
from transformers.tokenization_utils_base import BatchEncoding

class YouTubeDataset(Dataset):

    def __init__(self, splits, model_sample_rate = 16000, max_audio_len = 131072 * 2) -> None:
        super().__init__()
        self.id2idx = {}
        self.text = []
        self.audio = []
        self.label = []
        self.id = []
        self.tokens = None
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        CAPTION_DIR = "data/CaptionPunctuated/"
        AUDIO_DIR = "data/audio/"


        df = pd.read_csv("data/youtubeDataset.csv")
        cnt = 0
        for _, entry in df.iterrows():
            # dataEntry = {
            #     "id": entry["RECORD ID"],
            #     "label": int(entry["engagement_rate"])
            # }
            if entry["engagement_rate"] != 0:
                # print(entry["RECORD ID"])
                start_time = time.time()
                self.id.append(entry["RECORD ID"])
                if entry["engagement_rate"] >= splits[-1]:
                  self.label.append(len(splits))
                elif entry["engagement_rate"] < splits[0]:
                  self.label.append(0)
                else:
                  for i in range(len(splits)-1):
                    if entry["engagement_rate"] >= splits[i] and entry["engagement_rate"] < splits[i+1]:
                      self.label.append(i+1)
                      break
                with open(CAPTION_DIR+entry["RECORD ID"]+".txt",'r') as f:
                    self.text.append(f.read())

                # audio_file = AUDIO_DIR+entry["RECORD ID"]+".wav"
                # waveform, sample_rate = torchaudio.load(audio_file)
                # waveform = torchaudio.functional.resample(waveform, sample_rate, model_sample_rate)[:,:max_audio_len]
                # if waveform.shape[1] < max_audio_len:
                #     waveform = F.pad(waveform, pad=(0, max_audio_len - waveform.shape[1]), mode='constant', value=0)
                # self.audio.append(waveform)

                self.id2idx[entry["RECORD ID"]] = cnt
                # print(f"{cnt} - "+entry["RECORD ID"]+f": {time.time()-start_time} seconds.")

                cnt += 1

        self.label = torch.tensor(self.label,dtype=torch.int64)
        self.tokens = self.tokenizer(self.text, max_length = 512, padding=True, truncation=True, return_tensors="pt")
        # self.audio = torch.stack(self.audio, axis = 0)
        # torch.save(self.audio, "data/audio_tensors_long.pt")
        # self.audio = torch.load("data/audio_tensors_long.pt")
        # self.audio = self.audio.mean(dim=1)

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
                ), self.audio[index], self.label[index]
