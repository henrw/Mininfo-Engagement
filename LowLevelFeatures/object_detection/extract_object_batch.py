import torchvision
import torch
import argparse
import cv2
import os
import detect_utils
import numpy as np
from PIL import Image
import json

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                    help='minimum input size for the RetinaNet network')
parser.add_argument('-t', '--threshold', default=0.6, type=float,
                    help='minimum confidence score for detection')
args = vars(parser.parse_args())
print('USING:')
print(f"Minimum image size: {args['min_size']}")
print(f"Confidence threshold: {args['threshold']}")

# download or load the model from disk
model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, 
                                                            min_size=args['min_size'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the model onto the computation device
model.eval().to(device)

# load the test image
for vid_id in os.listdir('data/videos'):
    if not os.path.isdir('data/videos/'+vid_id):
        continue
    print(vid_id)
    scene_dict = {}
    try:
        scene_i = 1
        while True:
            for j in range(1,2):
                
                img_path = 'data/videos/'+vid_id + '/' + vid_id+'-Scene-'+str(scene_i).zfill(3)+'-0'+str(j)+'.jpg'
                image = Image.open(args['input']).convert('RGB')
                image_array = np.array(image)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                boxes, classes = detect_utils.predict(image, model, device, args['threshold'])

                # result = detect_utils.draw_boxes(boxes, classes, image_array)
                # cv2.imshow('Image', result)
                # cv2.waitKey(0)
                # save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}_t{int(args['threshold']*100)}"
                # cv2.imwrite(f"outputs/{save_name}.jpg", result)
                
                scene_dict[scene_i] = classes
            scene_i += 1
    except:
        json_object = json.dumps(scene_dict, indent=4)
        with open("data/features/scene/{}.json".format(vid_id), "w") as outfile:
            outfile.write(json_object)