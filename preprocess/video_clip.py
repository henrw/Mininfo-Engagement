import os
video_files = os.listdir("data/videos")
valid_ID = [f.strip() for f in open("data/valid_file.txt",'r').readlines()]
for video_file in video_files:
    if video_file.find('.') == -1:
        continue
    this_id = video_file.replace('.',' ').split()[0]
    if this_id in valid_ID:
        print("\""+video_file+"\"",end=" ")
        # new_dir = "data/videos/"+this_id
        # # os.makedirs(new_dir, exist_ok = True)
        # os.system("scenedetect -i data/videos/"+video_file+" detect-content list-scenes -o "+new_dir+" split-video -o "+new_dir+" --copy")