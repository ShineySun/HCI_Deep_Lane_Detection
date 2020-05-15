#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-12-19 02:09:53
# @Author  : Gefu Tang (tanggefu@gmail.com)
# @Link    : https://github.com/primetang/pylsd
# @Version : 0.0.1

import cv2
import numpy as np
import os
from pylsd.lsd import lsd
fullName = 'test_img1.jpg'
folder, imgName = os.path.split(fullName)
src = cv2.imread(fullName, cv2.IMREAD_COLOR)
tmp1 = cv2.imread("tmp.png",cv2.IMREAD_COLOR)
# cv2.waitKey(30000)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
lines = lsd(gray)
a = []
for i in range(lines.shape[0]):
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    width = lines[i, 4]
    dist = (pt1[0]-pt2[0])*(pt1[0]-pt2[0]) + (pt1[1]-pt2[1])*(pt1[1]-pt2[1])
    tmp = (i,dist,pt1,pt2)
    a.append(tmp)
    #print(dist)
b = sorted(a,key=lambda a : a[1],reverse=1)
cv2.line(tmp1, b[2][2], b[2][3], (255, 255, 255), 1)
print(b[0][2])



count = 0



cv2.imwrite(os.path.join(folder, 'cv2_' + imgName.split('.')[0] + '2.jpg'), tmp1)
