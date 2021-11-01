import cv2


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to_tuple(self):
        return self.x, self.y

    def __str__(self):
        return "x : %i | y : %i" % (self.x, self.y)

    def __repr__(self):
        return str(self)


class Seed(Point):
    def __init__(self, x, y, label):
        super().__init__(x, y)
        self.label = label

    def __str__(self):
        return "[%s] x : %i | y : %i" % (self.label, self.x, self.y)


def to_color(n):
    color_str = "{0:b}".format(n)
    size = len(color_str)
    r, g, b = (255, 255, 255)
    for i in range(size - 1, -1, -1):
        if int(color_str[i]):
            step = (size - 1 - i) % 3
            if step == 0:
                r = r // 2
            elif step == 1:
                g = g // 2
            elif step == 2:
                b = b // 2

    return b, g, r


def draw(im, seeds, disting=False):
    for seed in seeds:
        if disting:
            color = to_color(seed.label)
        else:
            color = (0, 0, 255)

        cv2.circle(im, (seed.y, seed.x), 2, color, -1)

    return im


def compact_label(seeds):
    new_seeds = []
    used_labels = []
    n_label = 0

    for seed in seeds:
        if seed.label in used_labels:
            new_label = used_labels.index(seed.label)
            new_seeds.append(Seed(seed.x, seed.y, new_label))
        else:
            n_label += 1
            new_seeds.append(Seed(seed.x, seed.y, n_label))

    return new_seeds, n_label
