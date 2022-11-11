import os 
with open("data/valid_file.txt",'r') as f:
    for line in f.readlines():
        filename = line.strip()
        os.system("python3 preprocess/gentle/align.py data/audio/"+filename+".wav data/CaptionPunctuated/"+filename+".txt -o data/text_audio_align/"+filename+".json")