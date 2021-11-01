import cv2
import numpy as np
from file_utils import get_dir
from seed import Seed
from seed import draw
from seed import compact_label


def reducer(im, seeds, thresh):
    size = len(seeds)
    temp = []

    is_full_merged = False
    loop_i = 0
    while not is_full_merged:
        is_full_merged = True
        i = 0
        while i < size - 1:
            curr = seeds[i]
            next_ = seeds[i + 1]

            if curr.y == next_.y and abs(int(im[curr.x, curr.y]) - int(im[next_.x, next_.y])) < thresh:
                x_ = int((curr.x + next_.x) / 2)
                y_ = curr.y
                if int(im[x_, y_]) != 255:
                    temp.append(Seed(x_, y_, curr.label))
                    is_full_merged = False
                    i += 2
                    continue

            if int(im[curr.to_tuple()]) != 255:
                temp.append(curr)
            i += 1

        seeds = temp
        size = len(seeds)
        temp = []
        loop_i += 1

    seeds, _ = compact_label(seeds)
    return seeds


def run(im, display_im):
    seeds_ = np.load("%s/../data/npz/initial.npz" % get_dir(__file__), allow_pickle=True)['seeds']
    thresh_ = 10

    seeds_ = reducer(im, seeds_, thresh_)
    reduced_im = draw(display_im, seeds_, True)
    np.savez("%s/../data/npz/reduced.npz" % get_dir(__file__), seeds=np.array(seeds_))
    cv2.imwrite("%s/../data/results/reduced.png" % get_dir(__file__), reduced_im)

    return reduced_im


if __name__ == "__main__":
    im_ = cv2.imread("%s/../data/results/gray.png" % get_dir(__file__), 0)
    display_im_ = cv2.imread("%s/../data/results/resized.png" % get_dir(__file__))

    reduced_im_ = run(im_, display_im_)

    while True:
        cv2.imshow("reduced_im", reduced_im_)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27=ESC
            break
    cv2.destroyAllWindows()
