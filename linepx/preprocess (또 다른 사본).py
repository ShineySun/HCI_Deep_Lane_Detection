import os
import cv2
import numpy as np
import scipy.io as sio
import pickle
import ref
from tqdm import tqdm
# 추가1-------------------------------------------------------S
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import json
# 추가1-------------------------------------------------------E0
def intx(x):
    return(int(x[0]), int(x[1]))

# 추가2-------------------------------------------------------S
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    # print(w,h)
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # print(buf.shape)
    buf = buf.reshape(h, w, 3)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    #    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf
# 추가2-------------------------------------------------------E

def processData(split):
    # Where to load raw images
    dataPath = ref.data_root / 'v1.1' / split
    imgnames = [x for x in os.listdir(dataPath) if x.endswith('.jpg')]
    # Where to load annotations
    annPath = ref.data_root / 'pointlines'
    # Where to store processed images
    outPath = ref.data_root / 'linepx' / 'processed'

    inpSize = ref.input_size # 320

    if not os.path.exists(outPath):
        os.makedirs(outPath)
# 추가3-------------------------------------------------------S
   # 1280 X 720
   # with open(r'/Users/chanwoo/Downloads/bdd-data-master/bdd-data-master/bdd100k/labels/bdd100k_labels_images_val.json', 'r') as f:
    with open(r'/home/foscar/Desktop/bdd/bdd100k/labels/bdd100k_labels_images_val.json', 'r') as f:
        photo_infos = json.load(f)
    for photo_info in photo_infos:
        # fig, ax = plt.subplots()
        img_location = os.path.abspath("/home/foscar/Desktop/bdd/bdd100k/all")
        img_location = img_location + "/" + photo_info['name']
        # print(img_location)
        img = cv2.imread(img_location, cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        # cv2.imshow('origin', img)
        arr = np.zeros((720, 1280, 3), dtype=np.uint8) + 255

        fig = plt.figure()
        ax = fig.add_subplot(111)
        lines_original = []
#         ax.imshow(arr)
        for labels in photo_info['labels']:
            if labels['category'] != 'lane':
                continue
            Path = mpath.Path
            path_data = []
            imgSize = np.array((w, h))
            for poly in labels['poly2d']:# /Users/chanwoo/Downloads/bdd-data-master/bdd-data-master/bdd100k/labels/bdd100k_labels_images_val.json
                for idx in range(len(poly['types'])):
                    if idx < len(poly['types']) - 1:
                        lines_original.append(poly['vertices'][idx] + poly['vertices'][idx + 1])
                    if idx == 0:
                        t = (Path.MOVETO, (poly['vertices'][idx][0], poly['vertices'][idx][1]))
                        path_data.append(t)
                    else:
                        a = Path.LINETO
                        if poly['types'][idx] == 'C':
                            a = Path.CURVE4
                        t = (a, (poly['vertices'][idx][0], poly['vertices'][idx][1]))
                        path_data.append(t)
            codes, verts = zip(*path_data)
            path = mpath.Path(verts, codes)
            if labels['attributes']['laneDirection'] == 'vertical':
                patch = mpatches.PathPatch(path, edgecolor='black', alpha= 1, fill=False)
            else:
                patch = mpatches.PathPatch(path, edgecolor='black', alpha= 1, fill=False)
            # print(patch.get_path())
            ax.add_patch(patch)
           # print(labels)
       # ax.imshow(img)
       # ax.axis('off')
        lines_original = np.array(lines_original)
#        plt.show()
        arr = fig2data(fig)
        bugcut = arr[80:401, 52:625]
        dst = ~(cv2.resize(bugcut, dsize=(1280, 720), interpolation=cv2.INTER_LINEAR))
        line = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#        cv2.imshow('arr2', line)
# 추가3-------------------------------------------------------E
'''
   for idx, in_ in enumerate(tqdm(imgnames)):
       # print("img {}".format(idx))
       img_path = str(dataPath / in_)
       img = cv2.imread(img_path)
       h, w, _ = img.shape
       line = np.zeros((inpSize, inpSize))
​
       with open(annPath / "{}{}".format(in_[:-4], ref.ext), 'rb') as f:
           target = pickle.load(f, encoding='latin1')
           points = target['points'] # 점들을 가지고 있는 리스트
           lines = target['lines'] # points에 대한 index를 가지고 있는 리스트
           lines_original = [] # 두 점을 [[x1, y1, x2, y2], .....]형태로 저장
           for i, j in lines:
               imgSize = np.array((w, h))
               start = np.array( points[i] ) * inpSize / imgSize
               end = np.array( points[j] ) * inpSize / imgSize
               lines_original.append(points[i] + points[j])# 두 점 더해준다
               dist = np.linalg.norm(end - start) / (inpSize * np.sqrt(2))
               line = cv2.line(line, intx(start), intx(end), 255 * dist, 2)
           lines_original = np.array(lines_original)
'''
        # 빈 어레이에 선만 그려준다

        save_imgname = outPath / "{}{}".format(in_[:-4], '_rgb.png')
        lineName = outPath / "{}{}".format(in_[:-4],'_line.png')
        lineOrigName = outPath / "{}{}".format(in_[:-4],'_line.mat')
        outImg = cv2.resize(img, (inpSize, inpSize))
        break

        sio.savemat(lineOrigName, {'lines':lines_original})
        cv2.imwrite(str(save_imgname), outImg)
        cv2.imwrite(str(lineName), line)
