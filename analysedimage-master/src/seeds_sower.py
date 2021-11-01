import cv2
import numpy as np
from file_utils import get_dir
from seed import Seed
from seed import draw
from image_editor import grid


def sow(dim, area_size):
    seeds = []
    for i in range(dim[0] // area_size):
        xd = i * area_size
        for j in range(dim[1] // area_size):
            yd = j * area_size
            seeds.append(Seed(int(yd + (area_size / 2)), int(xd + (area_size / 2)), (i * j)))
    return seeds


def run(im):
    area_size_ = 16
    seeds_ = sow(im.shape, area_size_)
    np.savez("%s/../data/npz/initial.npz" % get_dir(__file__), seeds=np.array(seeds_))
    gridded_im = grid(im, area_size_)
    sowed_im = draw(gridded_im.copy(), seeds_)

    cv2.imwrite("%s/../data/results/gridded.png" % get_dir(__file__), gridded_im)
    cv2.imwrite("%s/../data/results/sowed.png" % get_dir(__file__), sowed_im)

    return gridded_im, sowed_im


if __name__ == "__main__":
    im_ = cv2.imread("%s/../data/results/resized.png" % get_dir(__file__))
    gridded_im_, sowed_im_ = run(im_)

    while True:
        cv2.imshow("gridded_im", gridded_im_)
        cv2.imshow("sowed_im", sowed_im_)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27=ESC
            break
    cv2.destroyAllWindows()
