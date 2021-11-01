import cv2
from file_utils import get_dir


def resize(im, dim):
    return cv2.resize(im, dim, interpolation=cv2.INTER_AREA)


def resize_from_im(source, dest):
    dim = dest.shape
    resized = resize(source, (dim[1], dim[0]))
    return resized


def gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def crop(im):
    im[im == 255] = 1
    im[im == 0] = 255
    im[im == 1] = 0
    gray_im = gray(im)
    im_dim = gray_im.shape
    thresh = cv2.threshold(gray_im, 127, 255, cv2.THRESH_BINARY_INV)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_cnt_dim = 0
    max_cnt = None
    for i in range(0, len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        if im_dim[0]*im_dim[1] > w*h > max_cnt_dim:
            max_cnt_dim = w*h
            max_cnt = x, y, w, h

    x, y, w, h = max_cnt
    return im[y:y + h, x:x + w]


def grid(im, area_size):
    im_dim = im.shape
    for i in range(im_dim[0] // area_size):
        x = i * area_size
        for j in range(im_dim[1] // area_size):
            y = j * area_size
            cv2.rectangle(im, (y, x), (y + area_size, x + area_size), (255, 0, 0), 2)

    return im


def run(im):
    cropped_im = crop(im)
    cv2.imwrite("%s/../data/results/cropped.png" % get_dir(__file__), cropped_im)
    dim_ = (512, 512)
    resized_im = resize(cropped_im, dim_)
    cv2.imwrite("%s/../data/results/resized.png" % get_dir(__file__), resized_im)
    gray_im = gray(resized_im)
    cv2.imwrite("%s/../data/results/gray.png" % get_dir(__file__), gray_im)

    return cropped_im, resized_im, gray_im


if __name__ == "__main__":
    im_ = cv2.imread("%s/../data/results/no_bg.png" % get_dir(__file__))
    cropped_im_, resized_im_, gray_im_ = run(im_)

    while True:
        cv2.imshow("cropped_im", cropped_im_)
        cv2.imshow("resized_im", resized_im_)
        cv2.imshow("gray_im", gray_im_)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27=ESC
            break
    cv2.destroyAllWindows()
