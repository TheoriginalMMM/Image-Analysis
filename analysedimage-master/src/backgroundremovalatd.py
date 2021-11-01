import cv2 as cv
import numpy as np
from file_utils import get_dir


def resize(dst, img):
    width = img.shape[1]
    height = img.shape[0]
    dim = (width, height)
    resized = cv.resize(dst, dim, interpolation=cv.INTER_AREA)
    return resized


def postTraitement(fgmask, openingKernelSize, closingKernelSize, NbIteration, medianBlurKernelSize):
    kernel = np.ones((openingKernelSize, openingKernelSize))
    fgmask = cv.medianBlur(fgmask, medianBlurKernelSize)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    nbiteration = NbIteration
    size = (medianBlurKernelSize, medianBlurKernelSize)
    shape = cv.MORPH_ELLIPSE
    kernel = cv.getStructuringElement(shape, size)
    kernel = np.ones(size, np.uint8)
    for i in range(nbiteration):
        fgmask = cv.dilate(fgmask, kernel)
        fgmask = cv.erode(fgmask, kernel)

    kernel = np.ones((closingKernelSize, closingKernelSize))
    return cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)


def replaceBacground(bg, fgMASK, frame):
    background = resize(bg, frame)
    resultats = np.zeros(bg.shape)
    fgmask_inv = cv.bitwise_not(fgMASK)
    # use the masks to extract the relevant parts from FG and BG
    fgimg = cv.bitwise_and(frame, frame, mask=fgMASK)
    bgimg = cv.bitwise_and(background, background, mask=fgmask_inv)
    res = cv.add(bgimg, fgimg)
    return res


def SetNMixturesMOG(val):
    newH = cv.getTrackbarPos("history", "MOG2_PARAMETRES")
    newTh = cv.getTrackbarPos("thersold", "MOG2_PARAMETRES")
    backSubMOG2 = cv.createBackgroundSubtractorMOG2(newH, newTh, False)
    backSubMOG2.setNMixtures(val)
    print("NEW MOG WITH :", backSubMOG2.getNMixtures(), " MIXTURES")


def SetLearningRate(val):
    learning_rate = val
    #print("NEW VALUE OF THE LEARNING RATE", learning_rate)


def Set_HistoryMOG(val):
    newH = cv.getTrackbarPos("history", "MOG2_PARAMETRES")
    backSubMOG2.setHistory(newH)


def Set_HistoryKNN(val):
    newTh = cv.getTrackbarPos("Dist2Threshold", "KNN_PARAMETRES")
    newH = cv.getTrackbarPos("history", "KNN_PARAMETRES")
    backSubKNN = cv.createBackgroundSubtractorKNN(newH, newTh, False)


def Set_ThersoldMOG(val):
    print(val)
    backSubMOG2.setVarThreshold(val)


def Set_ThersoldKNN(val):
    backSubKNN.setDist2Threshold(val)


def Set_kNNSamples(val):
    backSubKNN.setkNNSamples(val)


def Set_Opening_Kernel_Size(val):
    openingKernelSize = val
    print(openingKernelSize)


def Set_Closing_Kernel_Size(val):
    closingKernelSize = val
    print(closingKernelSize)


def SetNbIterations(val):
    NbIteration = val
    print(NbIteration)


def SetMedianBlurKernelSize(val):
    medianBlurKernelSize = val
    print(medianBlurKernelSize)


# BOOLEAN TO DISABLE POST TRAITEMENT OF THE FOGROUND MASK
POST_TRAITEMENT = True
if __name__ == "__main__":
    # cv.createTrackbar("History length","panel",0,1000,initialiserModel())

    global learning_rate
    learning_rate = -1
    global backSubMOG2
    backSubMOG2 = cv.createBackgroundSubtractorMOG2(detectShadows=False)

    global backSubKNN
    backSubKNN = cv.createBackgroundSubtractorKNN(detectShadows=False)

    MOG2_PARAMETRES = np.zeros([10, 10], np.uint8)
    KNN_PARAMETRES = np.zeros([10, 10], np.uint8)
    cv.namedWindow('MOG2_PARAMETRES')
    cv.namedWindow("KNN_PARAMETRES")

    if (POST_TRAITEMENT):
        global openingKernelSize
        openingKernelSize = 5

        global closingKernelSize
        closingKernelSize = 10

        global NbIteration
        NbIteration = 1

        global medianBlurKernelSize
        medianBlurKernelSize = 3

        cv.namedWindow("POST_TRAITEMENT_PARAMETRES")
        cv.createTrackbar("Opening operation Kernel Size", "POST_TRAITEMENT_PARAMETRES", 1, 10, Set_Opening_Kernel_Size)
        cv.createTrackbar("Closing Kernel Size", "POST_TRAITEMENT_PARAMETRES", 1, 10, Set_Closing_Kernel_Size)
        cv.createTrackbar("Nb Iterations", "POST_TRAITEMENT_PARAMETRES", 1, 5, SetNbIterations)
        cv.createTrackbar("Median Blur Kernel Size", "POST_TRAITEMENT_PARAMETRES", 1, 10, SetMedianBlurKernelSize)

    cv.createTrackbar("history", "MOG2_PARAMETRES", 0, 10000, Set_HistoryMOG)
    cv.createTrackbar("thersold", "MOG2_PARAMETRES", 0, 50, Set_ThersoldMOG)
    cv.createTrackbar("Nb of mextures", "MOG2_PARAMETRES", 0, 5, SetNMixturesMOG)
    cv.createTrackbar("learning rate", "MOG2_PARAMETRES", 0, 1, SetLearningRate)

    cv.createTrackbar("History", "KNN_PARAMETRES", 0, 1000, Set_HistoryKNN)
    cv.createTrackbar("Dist2Threshold", "KNN_PARAMETRES", 0, 1000, Set_ThersoldKNN)
    cv.createTrackbar("History", "KNN_PARAMETRES", 0, 1000, Set_HistoryKNN)
    cv.createTrackbar("kNN Samples ", "KNN_PARAMETRES", 0, 9, Set_kNNSamples)

    # Initialisation des differents MODEL  #SANS DETECTER L'OMBRE DES OBJET POUR ACCELER
    backSubMOG2 = cv.createBackgroundSubtractorMOG2(detectShadows=False)
    backSubKNN = cv.createBackgroundSubtractorKNN(detectShadows=False)

    # On ouvre la caméra principale de la machine
    capture = cv.VideoCapture(0)
    # On lit l'image a utilisé pour remplacer le background
    BACKGROUND = cv.imread("%s/../data/img/bg.png" % get_dir(__file__))

    if not capture.isOpened():
        print("IMPOSSIBLE D'OUVRIR LA CAMÉRA ")

    while True:
        # On lis le frame acctuelle de l'image
        ret, frame = capture.read()
        if frame is None:
            print("probleme")
            break

        fgMaskMOG2 = backSubMOG2.apply(frame, learningRate=learning_rate)
        fgMaskKNN = backSubKNN.apply(frame, learningRate=learning_rate)

        resultatsMOG2 = replaceBacground(BACKGROUND, fgMaskMOG2, frame)
        resultatsKNN = replaceBacground(BACKGROUND, fgMaskKNN, frame)

        if (POST_TRAITEMENT):
            # On traite nos masques
            cv.imshow("FGMASK KNN", fgMaskKNN)
            cv.imshow("FGMASK MOG2", fgMaskMOG2)

            fgMaskMOG2MODIFIE = postTraitement(fgMaskMOG2, openingKernelSize, closingKernelSize, NbIteration,
                                               medianBlurKernelSize)
            fgMaskKNNMODIFIE = postTraitement(fgMaskKNN, openingKernelSize, closingKernelSize, NbIteration,
                                              medianBlurKernelSize)

            cv.imshow("FGMASK KNN TRAITER", fgMaskKNNMODIFIE)
            cv.imshow("FGMASK MOG2 TRAIRER", fgMaskMOG2MODIFIE)

            # On remplace le background
            # resultatsMOG=replaceBacground(BACKGROUND,fgMaskMOGMODIFIE,frame)
            resultatsMOG2 = replaceBacground(BACKGROUND, fgMaskMOG2MODIFIE, frame)
            resultatsKNN = replaceBacground(BACKGROUND, fgMaskKNNMODIFIE, frame)

        else:

            # On affiche les masques
            cv.imshow("FGMASK KNN", fgMaskKNN)
            cv.imshow("FGMASK MOG2", fgMaskMOG2)

        # ON AFFICHE LE RESULTATS FINAL
        cv.imshow("RESULTATS FINAL MOG2 ", resultatsMOG2)
        cv.imshow("RESULTATS FINAL KNN ", resultatsKNN)

        keyboard = cv.waitKey(30) & 0xFF
        if ord('q') == keyboard:
            cv.destroyAllWindows()
            break
