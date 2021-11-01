import cv2
import numpy as np
from file_utils import get_dir
from seed import Seed
from seed import Point
from seed import draw
from seed import compact_label


def get_gray_diff(img, current_point, tmp_point):
    return abs(int(img[current_point.x, current_point.y]) - int(img[tmp_point.x, tmp_point.y]))


def select_connects(is_8_conn):
    if is_8_conn:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects


def expand(im, seeds, thresh, is_8_conn=True):
    height, weight = im.shape
    seed_marks = np.zeros((height, weight), dtype=int)
    connects = select_connects(is_8_conn)
    merges = {}
    for seed in seeds:
        merges[seed.label] = 0

    def set_merge(min_, max_):
        if merges[max_] == 0 or merges[max_] > min_:
            merges[max_] = min_
        elif merges[max_] < min_:
            set_merge(merges[max_], min_)

        for i in merges:
            if merges[i] == max_ and merges[max_] != 0:
                merges[i] = merges[max_]

    seeds = seeds.tolist()
    while len(seeds) > 0:
        seed = seeds.pop(0)

        seed_marks[seed.x, seed.y] = seed.label
        for connect in connects:
            tmp_x = seed.x + connect.x
            tmp_y = seed.y + connect.y
            if tmp_x < 0 or tmp_y < 0 or tmp_x >= height or tmp_y >= weight:
                continue
            gray_diff = get_gray_diff(im, seed, Seed(tmp_x, tmp_y, seed.label))
            if int(im[seed.x, seed.y]) != 255 and seed_marks[tmp_x, tmp_y] != seed.label and gray_diff < thresh:
                if seed_marks[tmp_x, tmp_y] == 0:
                    seed_marks[tmp_x, tmp_y] = seed.label
                    seeds.append(Seed(tmp_x, tmp_y, seed.label))
                else:
                    m0 = min(seed.label, seed_marks[tmp_x, tmp_y])
                    m1 = max(seed.label, seed_marks[tmp_x, tmp_y])
                    set_merge(m0, m1)

    for (x, y), mark in np.ndenumerate(seed_marks):
        if mark != 0 and merges[mark] != 0:
            seed_marks[x, y] = merges[mark]

    new_seeds = []
    for (x, y), mark in np.ndenumerate(seed_marks):
        if mark != 0:
            new_seeds.append(Seed(x, y, mark))
    compact_label(new_seeds)
    return np.array(new_seeds)


def run(im, display_im):
    seeds_ = np.load("%s/../data/npz/reduced.npz" % get_dir(__file__), allow_pickle=True)['seeds']
    thresh_ = 4

    seeds_ = expand(im, seeds_, thresh_)
    expanded_im = draw(display_im, seeds_, True)
    np.savez("%s/../data/npz/expanded.npz" % get_dir(__file__), seeds=np.array(seeds_))
    cv2.imwrite("%s/../data/results/expanded.png" % get_dir(__file__), expanded_im)

    return expanded_im


if __name__ == "__main__":
    im_ = cv2.imread("%s/../data/results/gray.png" % get_dir(__file__), 0)
    display_im_ = cv2.imread("%s/../data/results/resized.png" % get_dir(__file__))
    expanded_im_ = run(im_, display_im_)

    while True:
        cv2.imshow("expanded_im", expanded_im_)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27=ESC
            break
    cv2.destroyAllWindows()
