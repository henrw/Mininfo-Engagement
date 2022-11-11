import os
import json
import time
import datetime

def merge_scene_prediction(scenes_from_three_frame, threshold_sum = 1, threshold_diff = 0.5):
    try:
        new_dict = {}
        if isinstance(scenes_from_three_frame, dict):
            scenes_from_three_frame = list(scenes_from_three_frame.values())
        for scene in scenes_from_three_frame:
            for key, prob in scene.items():
                new_dict[key] = (new_dict[key]+prob if key in new_dict.keys() else prob)
        predictions = [(k, v) for k, v in sorted(new_dict.items(), key=lambda item: item[1], reverse=True)]
        # print(predictions[0][1] - predictions[1][1])
        if predictions[0][1] >= threshold_sum and (predictions[0][1] - predictions[1][1]) > threshold_diff:
            return predictions[0]
        return None
    except:
        return None

def output_formatting(scene, type="at"):
    if type == "at":
        return f"at {scene}"
    
if __name__ == "__main__":
    with open("data/valid_file.txt", "r") as f:
        file_ids = [file_id.strip() for file_id in f.readlines()]

    text_time_reference_dir = "data/text_audio_align/"
    feature_dir = "data/features/scene/"
    clips_dir = "data/videos/"
    output_dir = "data/tmp/transcripts_with_scenes/"
    os.makedirs(output_dir,exist_ok=True)

    for file_id in file_ids:
        print(f"---{file_id}---")

        with open(text_time_reference_dir+file_id+'.json', 'r') as align_f:
            data = json.load(align_f)
            text = data["transcript"]
            words = data["words"]

        with open(clips_dir+file_id+"/"+file_id+"-Scenes.csv") as clip_f:
            timestamps = clip_f.readline().strip().replace(',',' ').split()[2:]
            timestamps.insert(0,"00:00:00")
        
        scenes = []
        with open(feature_dir+file_id+'.json', 'r') as feature_f:
            scene_clips = list(json.load(feature_f).values())
            for scene_per_clip, timestamp in zip(scene_clips,timestamps):
                h, m, s = [float(s) for s in timestamp.replace(":"," ").split()]
                start_time = h*3600+m*60+s
                scene_pred = merge_scene_prediction(scene_per_clip)
                if scene_pred:
                    scenes.append((scene_pred[0], start_time))
        
        new_transcript = ""
        while scenes or words:
            if len(scenes) == 0:
                new_transcript += text
                words = []
            elif len(words) == 0:
                new_transcript += f" {output_formatting(scenes[0][0])}"
                scenes.pop(0)
            elif words[0]["case"] == "not-found-in-audio":
                words.pop(0)
            elif float(words[0]["end"]) > scenes[0][1]:
                if not new_transcript:
                    new_transcript = f"{output_formatting(scenes[0][0])} "
                else:
                    new_transcript += f" {output_formatting(scenes[0][0])}"
                scenes.pop(0)
            else:
                pos = text.find(words[0]["word"])
                assert(pos!=-1)
                idx = pos+len(words[0]["word"])
                new_transcript, text = new_transcript+text[:idx], text[idx:]
                words.pop(0)

        with open(output_dir+"/"+file_id+".txt", 'w') as out_f:
            out_f.write(new_transcript)
                
            


