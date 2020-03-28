### 받아온 영상을 320 × 320 으로 변환
import os
import cv2

# cnt = 0

for root, dirs, files in os.walk('/home/foscar/Desktop/bdd/bdd100k/line1920val'):
    for fname in files:
        # cnt += 1
        full_fname = os.path.join(root, fname)
        if fname == '.DS_Store':
            continue
        img = cv2.imread(full_fname, cv2.IMREAD_COLOR)
        print(fname)

        # cv2.imshow('origin', img)
        area = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
        # cubic = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)
        # lanc = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LANCZOS4)
        # nearest = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)
        # linear = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('origin', img)
        # cv2.imshow('AREA', area)
        # cv2.imshow('CUBIC', cubic)
        # cv2.imshow('LANC', lanc)
        # cv2.imshow('NEAREST', nearest)
        # cv2.imshow('LINEAR', linear)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        name = '/home/foscar/Desktop/bdd/bdd100k/line320val/' + fname[:-4] + '_line.png'
        cv2.imwrite(name, area)
        # cv2.waitKey(0)
        # if cnt > 10 :
        #     break
