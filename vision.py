import json
import os
import cv2
import numpy as np
result_path = r"result2.json"
f = open(result_path,'r')
results = json.load(f)
f.close()
img_path = r"D:\dataset\pandiqi\cow_wight_1\RGB"
out_path = r"D:\dataset\pandiqi\cow_wight_1\cpn_res_v2"
imgs = os.listdir(img_path)
for i,result in enumerate(results):
    points = result['keypoints']
    name = result['image_id']
    for j in range(len(points)):
        points[j] = int(points[j])
    #points = np.asarray(points,dtype='uint8')
    img  = cv2.imread(os.path.join(img_path,name))
    point_list = [(points[0],points[1]),(points[3],points[4]),(points[6],points[7]),(points[9],points[10])]
    # point_list = [(537,286)]
    for point in point_list:
        cv2.circle(img, point, 3, (0, 0, 255), 4)
    cv2.imwrite(os.path.join(out_path,name), img)
    #cv2.waitKey(10000)
