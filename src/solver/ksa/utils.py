import numpy as np


class Box:
    n_boxes = 0
    z_ideal_bar = None

    IDEAL = 0
    LOWER = 1

    def __init__(self, l=None, u=None, cal_area_by=0):
        assert (l <= u).all(), "Invalid box!"
        self.id = Box.n_boxes
        Box.n_boxes += 1

        self.l = l
        self.u = u
        if cal_area_by == Box.IDEAL:
            self.A = np.log(np.prod(np.subtract(self.u, Box.z_ideal_bar)))
        elif cal_area_by == Box.LOWER:
            self.A = np.prod(np.subtract(self.u, self.l))
        else:
            raise ValueError("Invalid area calculation method!")

    @classmethod
    def reset(cls, z_ideal_bar=None):
        cls.n_boxes = 0
        cls.z_ideal_bar = None if z_ideal_bar is None else np.asarray(z_ideal_bar)


class Efficient:
    def __init__(self, x, y, time=None):
        self.x = x
        self.y = y
        self.time = time
