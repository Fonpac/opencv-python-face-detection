import cv2 as cv
import numpy as np
from glob import glob
import argparse
import os

face_cascade = cv.CascadeClassifier('.\HaarCascade\haarcascade_frontalface_alt.xml')

def parseArgs():
    """Parse the arguments using `argparse.ArgumentParser`"""
    parser = argparse.ArgumentParser(
        description='Face recognition script'
    )

    parser.add_argument(
        "images", 
        default=".\images\*.jpg",
        help="Folder containing the images",
        nargs="?"
    )

    parser.add_argument(
        "-o", 
        "--output", 
        default=".\outputs",
        help="Output directory for the result image",
        nargs="?"
    )

    return parser.parse_args()

def blurFaces(img, faces):
    maskShape = (img.shape[0], img.shape[1], 1)
    mask = np.full(maskShape, 0, dtype=np.uint8)
    tempImg = img.copy()

    for (x, y, w, h) in faces:
        tempImg[y: y+h, x: x+w] = cv.blur(tempImg [y: y+h, x: x+w], (23,23))
        cv.circle(tempImg, ( int((x + x + w) / 2 ), int((y + y + h) / 2 )), int (h / 2), (255), 3)
        cv.circle(mask, ( int((x + x + w) / 2 ), int((y + y + h) / 2 )), int (h / 2), (255), -1)

    mask_inv = cv.bitwise_not(mask)
    img1_bg = cv.bitwise_and(img,img,mask = mask_inv)
    img2_fg = cv.bitwise_and(tempImg,tempImg,mask = mask)
    dst = cv.add(img1_bg,img2_fg)

    return dst

def main():
    args = parseArgs()

    imgsPath = args.images
    outputPath = args.output

    imgs = []

    for file in glob(imgsPath):
        img = cv.imread(file, cv.IMREAD_UNCHANGED)
        fileName = file.split('\\')[-1]
        obj = { "fileName": fileName, "img": img }
        imgs.append(obj)

    for img in imgs:
        grayImg = cv.cvtColor(img['img'], cv.COLOR_BGR2GRAY)
        facesAlt = face_cascade.detectMultiScale(grayImg, 1.1, 3)
        
        dst = blurFaces(img['img'], facesAlt)

        output = os.path.join(outputPath, img["fileName"])

        cv.imwrite(output,dst)


if __name__ == "__main__":
    main()
