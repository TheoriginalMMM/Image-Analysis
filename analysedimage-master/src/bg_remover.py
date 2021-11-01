import cv2
import numpy as np
from file_utils import get_dir
from image_editor import resize_from_im
from image_editor import gray


def remove_bg(im, bg):
    bg = resize_from_im(bg, im)

    # create a mask
    diff1 = cv2.subtract(im, bg)
    diff2 = cv2.subtract(bg, im)
    diff = diff1 + diff2
    diff[abs(diff) < 22.0] = 0
    gray_im = gray(diff.astype(np.uint8))
    gray_im[np.abs(gray_im) < 10] = 0
    fg_mask = gray_im.astype(np.uint8)
    fg_mask[fg_mask > 0] = 255

    # use the masks to extract the relevant parts from FG and BG
    fg_im = cv2.bitwise_and(im, im, mask=fg_mask)

    return fg_im


def run(im, bg_im):
    no_bg_im = remove_bg(im, bg_im)
    cv2.imwrite("%s/../data/results/no_bg.png" % get_dir(__file__), no_bg_im)

    return no_bg_im


if __name__ == "__main__":
    im_ = cv2.imread("%s/../data/img/body_1_1.png" % get_dir(__file__))
    bg_im_ = cv2.imread("%s/../data/img/bg.png" % get_dir(__file__))

    im_no_bg_ = run(im_, bg_im_)

    while True:
        cv2.imshow("img_no_bg", im_no_bg_)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27=ESC
            break
    cv2.destroyAllWindows()
