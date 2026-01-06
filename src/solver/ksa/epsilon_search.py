import copy
import time
from collections import deque

import numpy as np

from .utils import Box


class EpsilonSearch:
    """Search the criterion space to enumerate nondominated solutions."""

    def __init__(self, moop):
        self.moop = moop
        self.L = {}
        self.Z_n = deque()
        self.X_e = deque()
        self.timer = deque()
        self.n_evaluations = 0

        self.MAX_VOLUME_IDEAL = 0
        self.MAX_VOLUME_LOWER = 1
        self.FROM_TOP = 2
        self.RANDOM = 3

    def _is_new_nondominated_solution(self, z):
        if len(self.Z_n) == 0:
            return True

        Z_n = np.asarray(self.Z_n)
        for i in range(self.moop.n_objectives):
            if z[i] in Z_n[:, i]:
                Z_n = np.asarray(
                    [Z_n[j, :] for j in range(Z_n.shape[0]) if Z_n[j, i] == z[i]]
                )
            else:
                return True

        return False

    def _update_list(self, z_bar):
        new_boxes = {}
        del_ids = []
        for b_parent_id, b_parent in self.L.items():
            T = {b_parent_id: b_parent}
            for j in range(self.moop.n_objectives - 1):
                if b_parent.l[j] < z_bar[j] and b_parent.u[j] > z_bar[j]:
                    _T = {}
                    for b_child in T.values():
                        new_u = copy.deepcopy(b_child.u)
                        new_u[j] = z_bar[j]
                        b1 = Box(l=b_child.l, u=new_u)
                        _T[b1.id] = b1

                        new_l = copy.deepcopy(b_child.l)
                        new_l[j] = z_bar[j]
                        b2 = Box(l=new_l, u=b_child.u)
                        _T[b2.id] = b2

                    T = _T.copy()

            if len(T.keys()) > 1:
                new_boxes.update(T)
                del_ids.append(b_parent_id)

        for del_id in del_ids:
            del self.L[del_id]
        self.L.update(new_boxes)

    def _remove_rectangle(self, l=None, u=None):
        assert l.shape[0] == u.shape[0] and (l <= u).all()

        del_ids = []
        for b_idx, b in self.L.items():
            if (b.l >= l).all() and (b.u <= u).all():
                del_ids.append(b_idx)

        for del_id in del_ids:
            del self.L[del_id]

    def _select(self, criterion, window=None, select=None):
        if criterion == self.MAX_VOLUME_IDEAL:
            mv, mv_index = 0, -1
            for box_id, box in self.L.items():
                if mv < box.A:
                    mv = box.A
                    mv_index = box_id

            return mv_index, self.L[mv_index]

        elif criterion == self.FROM_TOP:
            pass
        elif criterion == self.MAX_VOLUME_LOWER:
            pass
        elif criterion == self.RANDOM:
            pass
        else:
            raise ValueError("Invalid rectangle selection criterion!")

    def run(self, time_limit=None):
        z_ideal, z_upper = self.moop.get_bounds()
        if z_ideal is None:
            return

        z_ideal_bar = z_ideal[self.moop.mask]
        z_upper_bar = z_upper[self.moop.mask]
        Box.reset(z_ideal_bar)

        self.moop.initialize_two_stage_model()
        b = Box(l=z_ideal_bar, u=z_upper_bar)
        self.L[b.id] = b

        time_limit = None if time_limit is None else float(time_limit)
        start_time = time.time()
        while len(self.L.keys()):
            if time_limit is not None and time.time() - start_time >= time_limit:
                print("Time limit reached; stopping early.")
                break
            sel_id, sel_box = self._select(self.MAX_VOLUME_IDEAL)
            self.n_evaluations += 1

            x, z = self.moop.get_solution(epsilon=sel_box.u)
            if x is not None:
                z_bar = z[self.moop.mask]
                if self._is_new_nondominated_solution(z):
                    self.timer.append(time.time() - start_time)
                    self.Z_n.append(z)
                    self.X_e.append(x)
                    self._update_list(z_bar)
                self._remove_rectangle(l=z_bar, u=sel_box.u)
            else:
                self._remove_rectangle(l=z_ideal_bar, u=sel_box.u)

        self.timer.append(time.time() - start_time)
