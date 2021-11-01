from backgroundremovalatd import postTraitement, Set_Opening_Kernel_Size, Set_Closing_Kernel_Size, \
    SetNbIterations, SetMedianBlurKernelSize
from file_utils import get_dir
import cv2
import numpy as np
import sys


def Set_DIFFERENCE_THERSOLD(val):
    DIFFTHERS = val


def Set_GRAY_THERSOLD(val):
    GRAYTHERS = val


def resize(dst, img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (width, height)
    resized = cv2.resize(dst, dim, interpolation=cv2.INTER_AREA)
    return resized


# BOOLEAN TO DISABLE POST TRAITEMENT OF THE FOGROUND MASK
POST_TRAITEMENT = True
if __name__ == "__main__":
    video = cv2.VideoCapture(0)
    bg = cv2.imread("%s/../data/img/bg.png" % get_dir(__file__))
    success, ref_img = video.read()
    flag = 0
    success, img = video.read()
    global DIFFTHERS
    DIFFTHERS = 13.0
    if (POST_TRAITEMENT):
        global openingKernelSize
        openingKernelSize = 5

        global closingKernelSize
        closingKernelSize = 10

        global NbIteration
        NbIteration = 1

        global medianBlurKernelSize
        medianBlurKernelSize = 3

        cv2.namedWindow("POST_TRAITEMENT_PARAMETRES")
        cv2.createTrackbar("Opening operation Kernel Size", "POST_TRAITEMENT_PARAMETRES", 1, 10,
                           Set_Opening_Kernel_Size)
        cv2.createTrackbar("Closing Kernel Size", "POST_TRAITEMENT_PARAMETRES", 1, 10, Set_Closing_Kernel_Size)
        cv2.createTrackbar("Nb Iterations", "POST_TRAITEMENT_PARAMETRES", 1, 5, SetNbIterations)
        cv2.createTrackbar("Median Blur Kernel Size", "POST_TRAITEMENT_PARAMETRES", 1, 10, SetMedianBlurKernelSize)

    global GRAYTHERS
    GRAYTHERS = 10
    cv2.namedWindow("PARAMETRES")
    cv2.createTrackbar("DIFFERENCE THERSOLD", "PARAMETRES", 1, 50, Set_DIFFERENCE_THERSOLD)
    cv2.createTrackbar("GRAY THERSOLD", "PARAMETRES", 1, 50, Set_GRAY_THERSOLD)
    while (1):
        success, img = video.read()
        # success2, bg = oceanVideo.read()
        bg = resize(bg, img)
        if flag == 0:
            ref_img = img
        # create a mask
        diff1 = cv2.subtract(img, ref_img)
        diff2 = cv2.subtract(ref_img, img)
        diff = diff1 + diff2
        diff[abs(diff) < DIFFTHERS] = 0
        gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray[np.abs(gray) < GRAYTHERS] = 0
        fgmask = gray.astype(np.uint8)
        fgmask[fgmask > 0] = 255
        cv2.imshow("Forground MASK", fgmask)
        # invert the mask
        if (POST_TRAITEMENT):
            fgmask = postTraitement(fgmask, openingKernelSize, closingKernelSize, NbIteration, medianBlurKernelSize)
            cv2.imshow("FGMASK TRAITE", fgmask)

        fgmask_inv = cv2.bitwise_not(fgmask)
        # use the masks to extract the relevant parts from FG and BG
        fgimg = cv2.bitwise_and(img, img, mask=fgmask)
        bgimg = cv2.bitwise_and(bg, bg, mask=fgmask_inv)
        # combine both the BG and the FG images
        dst = cv2.add(bgimg, fgimg)
        cv2.imshow('Background Removal', dst)
        key = cv2.waitKey(5) & 0xFF
        if ord('q') == key:
            break
        elif ord('d') == key:
            flag = 1
            print("Background Captured")
        elif ord('r') == key:
            flag = 0
            print("Ready to Capture new Background")

    cv2.destroyAllWindows()
    video.release()
