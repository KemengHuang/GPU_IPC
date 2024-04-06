import cv2
import os
from tqdm import tqdm
fsp = 30
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
 
 
n = 0
video_path = 'video.mp4'
img_path = "saveScreen"
list_image = os.listdir(img_path)
list_image.sort()
 
 
list_image = [os.path.join(img_path,x) for x in list_image]
width = cv2.imread(list_image[0]).shape[1]
heighth = cv2.imread(list_image[0]).shape[0]
 
video_out = cv2.VideoWriter(video_path, fourcc, fsp, (width,heighth))
print(len(list_image))
 
count = 0
for i in tqdm(range(len(list_image))):
     #if i == 0:
     #     continue
     if os.path.exists(list_image[i]):
          frame = cv2.imread(list_image[i])
          video_out.write(frame)
          count += 1
 
print('cout',count)
 
video_out.release()