import cv2
import numpy as np
import os

files = [
    'C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\warehouse_parts\\test\\Part001',
    'C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\warehouse_parts\\test\\Part002',
    'C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\warehouse_parts\\train\\Part001',
    'C:\\Users\\boyko\\OneDrive - University of Waterloo\\coding\\data\\warehouse_parts\\train\\Part002'
]

for file in files:
    dest = os.path.join(file, "processed")
    if not os.path.exists(dest):
        os.mkdir(dest)
    for image in os.listdir(file):
        if os.path.isfile(os.path.join(file, image)):
            img = cv2.imread(os.path.join(file, image))
            if img is None:
                print("unable to read image: {}".format(os.path.join(file, image)))
                exit(1)
            ## (1) Convert to gray, and threshold
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            th, threshed = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
            ## (2) Morph-op to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
            morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
            cv2.imshow('c', morphed.astype(np.uint8))
            ## (3) Find the max-area contour
            _, cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = sorted(cnts, key=cv2.contourArea)[-1]

            ## (4) Crop and save it
            x,y,w,h = cv2.boundingRect(cnt)
            dst = img[y:y+h, x:x+w]
            cv2.imshow('d', dst)
            cv2.waitKey(100)

            cv2.imwrite(os.path.join(dest, image), dst)