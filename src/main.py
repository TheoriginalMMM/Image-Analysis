import sys
import cv2
import bg_remover
import image_editor
import seeds_sower
import seeds_reducer
import seeds_expander
from file_utils import get_dir


class CmdException(Exception):
    pass


if __name__ == "__main__":
    body_path = "%s/../data/img/body_1_1.png" % get_dir(__file__)
    bg_path = "%s/../data/img/bg.png" % get_dir(__file__)

    cmd = -1
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "--body":
            if cmd == 0:
                raise CmdException("Your command '--body' need the filename of the body image")
            if cmd == 1:
                raise CmdException("Your command '--bg' need the filename of the background image")
            cmd = 0
        elif sys.argv[i] == "--bg":
            if cmd == 0:
                raise CmdException("Your command '--body' need the filename of the body image")
            if cmd == 1:
                raise CmdException("Your command '--bg' need the filename of the background image")
            cmd = 1
        else:
            if cmd == -1:
                raise CmdException("Bad command '%s' !" % sys.argv[i])
            elif cmd == 0:
                body_path = sys.argv[i]
                cmd = -1
            elif cmd == 1:
                bg_path = sys.argv[i]
                cmd = -1

    if cmd == 0:
        raise CmdException("Your command '--body' need the filename of the body image")
    if cmd == 1:
        raise CmdException("Your command '--bg' need the filename of the background image")

    body_im = cv2.imread(body_path)
    bg_im = cv2.imread(bg_path)
    no_bg_im = bg_remover.run(body_im.copy(), bg_im.copy())
    cropped_im, resized_im, gray_im = image_editor.run(no_bg_im.copy())
    gridded_im, sowed_im = seeds_sower.run(resized_im.copy())
    reduced_im = seeds_reducer.run(gray_im.copy(), resized_im.copy())
    expanded_im = seeds_expander.run(gray_im.copy(), resized_im.copy())

    while True:
        cv2.imshow("Region Growing", expanded_im)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27=ESC
            break
    cv2.destroyAllWindows()
