#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import numpy as np
import cv2
import json
import sys
cur_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_dir, 'MSCOCO', 'PythonAPI'))
#from pycocotools.coco import COCO

class COCOJoints(object):
    def __init__(self):
        self.kp_names = ['lu','ru','ld','rd']
        self.max_num_joints = 4
        self.color = np.random.randint(0, 256, (self.max_num_joints, 3))
        self.mpi = []
        self.test_mpi = []
        label_path = "/export/pigfarm/pandiqi/code/tf-cpn-master/data/cow/Annotation"
        img_path = "/export/pigfarm/pandiqi/code/tf-cpn-master/data/cow/Images"
        labels = os.listdir(label_path)
        aid=0
        for label in labels:
            f = open(os.path.join(label_path,label),'r')
            dic = json.load(f)
            f.close()
            print(dic)
            shapes = dic['shapes']
            bbox=[0]*4
            joints=[0]*12
            for shape in shapes:
                if shape['label']=='rect':
                    points = shape['points']
                    bbox[0] = points[0][0]
                    bbox[1] = points[0][1]
                    bbox[2] = points[1][0]-points[0][0]
                    bbox[3] = points[1][1]-points[0][1]
                if shape['label']=='lu':
                    points = shape['points']
                    joints[0] = points[0][0]
                    joints[1] = points[0][1]
                    joints[2] = 2#COCO rules :0.none 1.unvisable 2.visable ,in our work, joints in our pic are all visable
                if shape['label'] == 'ru':
                    points = shape['points']
                    joints[3] = points[0][0]
                    joints[4] = points[0][1]
                    joints[5] = 2  # COCO rules :0.none 1.unvisable 2.visable ,in our work, joints in our pic are all visable
                if shape['label']=='ld':
                    points = shape['points']
                    joints[6] = points[0][0]
                    joints[7] = points[0][1]
                    joints[8] = 2#COCO rules :0.none 1.unvisable 2.visable ,in our work, joints in our pic are all visable
                if shape['label']=='rd':
                    points = shape['points']
                    joints[9] = points[0][0]
                    joints[10] = points[0][1]
                    joints[11] = 2#COCO rules :0.none 1.unvisable 2.visable ,in our work, joints in our pic are all visable
            rect = np.array([0, 0, 1, 1], np.int32)
            imgname = os.path.join(img_path,label.replace(".json",".png"))
            aid+=1
            cowData = dict(aid=aid, joints=joints, imgpath=imgname, headRect=rect, bbox=bbox, imgid=aid,
                             segmentation=[[]])
            self.mpi.append(cowData)



    def load_data(self, min_kps=1):
        mpi = [i for i in self.mpi if np.sum(np.array(i['joints'], copy=False)[2::3] > 0) >= min_kps]
        return mpi, self.test_mpi

if __name__ == '__main__':
    coco_joints = COCOJoints()
    train, _ = coco_joints.load_data(min_kps=1)
    print(train[0])
    from IPython import embed; embed()
