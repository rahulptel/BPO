import copy
import pickle as pkl
import time
from collections import deque

import numpy as np

from .utils import Box


class EpsilonSearch:
    """Perform search in the criterion space to enumerate the nondominated
    solutions of a multi-objective optimization problem.

    Notes
    -----
    Python + Gurobi implementation [1]_.

    References
    ----------
    ..[1] Kirlik, G., & Sayın, S. (2014). A new algorithm for generating all nondominated solutions of
    multiobjective discrete optimization problems. European Journal of Operational Research, 232(3), 479-488.

    Attributes
    ----------
    moop: child object of .moop.problem.Problem
        A multi-objective optimization problem instance
    L: deque
        List of partitions of the criterion space
    Z_n: deque
        List of nondominated solutions

    Methods
    -------
    _is_new_nondominated_solution(z)
        Check if the new nondominated solution is already found
    _update_list(z_bar)
        Partition the criterion space based on the projected nondominated solution
    _remove_rectangle(l, u)
        Remove rectangles which lie in the imaginary rectangle R(l, u)
    _select(criterion, window=None, select=None)
        A region of the criterion space to search for new nondominated solution
    log()
        Log the results
    run()
        Begin an exhuastive search over the citerion space to list
    """

    def __init__(self, moop, save_path):
        self.moop = moop
        self.save_path = save_path
        self.L = {}
        self.Z_n = deque()
        self.X_e = deque()
        self.timer = deque()

        self.MAX_VOLUME_IDEAL = 0
        self.MAX_VOLUME_LOWER = 1
        self.FROM_TOP = 2
        self.RANDOM = 3

    def _is_new_nondominated_solution(self, z):
        """Check if the new nondominated solution is already found

        Parameters
        ----------
        z:
            New nondominated solution

        Returns
        -------
        bool:
            True, if the nondominated solution is not present in the current set
            False, otherwise
        """
        if len(self.Z_n) == 0:
            return True

        Z_n = np.asarray(self.Z_n)
        for i in range(self.moop.n_objectives):
            if z[i] in Z_n[:, i]:
                Z_n = np.asarray([Z_n[j, :] for j in range(Z_n.shape[0]) if Z_n[j, i] == z[i]])
            else:
                return True

        return False

    def _update_list(self, z_bar):
        """Partition the search space based on z_bar

        Parameters
        ----------
        z_bar:
            Projected nondominated solution
        Returns
        -------
        """
        new_boxes = {}
        del_ids = []
        for b_parent_id, b_parent in self.L.items():
            # print(b_parent.l, b_parent.u)
            T = {b_parent_id: b_parent}
            for j in range(self.moop.n_objectives - 1):
                # print('\t', b_parent.l[j], z_bar[j], b_parent.u[j])
                if b_parent.l[j] < z_bar[j] and b_parent.u[j] > z_bar[j]:
                    #                     print('\t True')
                    #                     print('\t len(T) = ', len(T))
                    # Split all the child rectangles in T
                    _T = {}
                    for b_child in T.values():
                        # make a copy and update upper bound on jth index
                        # new_u = b_child.u[:]
                        new_u = copy.deepcopy(b_child.u)
                        new_u[j] = z_bar[j]
                        # print('\t\t1. ', b_child.l, new_u)
                        b1 = Box(l=b_child.l, u=new_u)
                        _T[b1.id] = b1

                        # make a copy and update lower bound on jth index
                        new_l = copy.deepcopy(b_child.l)
                        new_l[j] = z_bar[j]
                        # print('\t\t2. ', new_l, b_child.u)
                        b2 = Box(l=new_l, u=b_child.u)
                        _T[b2.id] = b2

                    T = _T.copy()

            if len(T.keys()) > 1:
                # b_parent was partitioned
                new_boxes.update(T)
                del_ids.append(b_parent_id)

        # Update L
        for del_id in del_ids:
            del self.L[del_id]
        self.L.update(new_boxes)

    def _remove_rectangle(self, l=None, u=None):
        """Remove boxes from the list L which are included in the rectangle R(l, u)

        Args
        ----
        l: list of int
            Lower vertex of the rectangle
        u: list of int
            Upper vertex of the rectangle
        """
        assert l.shape[0] == u.shape[0] and (l <= u).all()

        # Find boxes not to remove
        del_ids = []
        for b_idx, b in self.L.items():
            # print(b_idx, b.l, ' >= ',l, ' && ', b.u, ' <= ' , u)
            if (b.l >= l).all() and (b.u <= u).all():
                # b not \in R(l, u)
                del_ids.append(b_idx)

        # Delete boxes
        for del_id in del_ids:
            del self.L[del_id]

    def _select(self, criterion, window=None, select=None):
        """Select rectangle based on a given criterion

        criterion: string
            Criterion used to select the rectangles. Listed below
            are the valid options for the same.

            'MAX_VOLUME_IDEAL': Rectangle with largest area as defined
                by its upper vertex and ideal lower vertex

            'FROM_TOP': Rectangle selected from `window` rectangles,
                that are sorted by decreasing `max_volume_ideal`

            'MAX_VOLUME_LOWER': Rectangle with largest area as defined
                by its upper vertex and lower vertex

            'RANDOM':

        window: int

        select: int

        """
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
            raise ValueError('Invalid rectangle selection criterion!')

    def log(self):
        f = self.moop.filename.stem + '.pkl'
        f = self.save_path / f
        pkl.dump({'Z_n': self.Z_n,
                  'X_e': self.X_e,
                  'timer': self.timer},
                 open(f, 'wb'))

    def run(self):
        z_ideal, z_upper = self.moop.get_bounds()
        if z_ideal is None:
            return

        # Project upper and lower bounds
        z_ideal_bar = z_ideal[self.moop.mask]
        z_upper_bar = z_upper[self.moop.mask]
        Box.z_ideal_bar = z_ideal_bar
        #         print(z_ideal_bar)

        self.moop.initialize_two_stage_model()
        # Create box and initialize list
        b = Box(l=z_ideal_bar, u=z_upper_bar)
        self.L[b.id] = b
        i = 0
        start_time = time.time()
        while len(self.L.keys()):
            # print(i, len(self.L.keys()))

            sel_id, sel_box = self._select(self.MAX_VOLUME_IDEAL)
            # print("Sel", sel_id)

            x, z = self.moop.get_solution(epsilon=sel_box.u)
            if x is not None:
                # Project nondominated solution
                z_bar = z[self.moop.mask]
                if self._is_new_nondominated_solution(z):
                    self.timer.append(time.time() - start_time)
                    self.Z_n.append(z)
                    self.X_e.append(x)
                    self._update_list(z_bar)
                self._remove_rectangle(l=z_bar, u=sel_box.u)
            else:
                self._remove_rectangle(l=z_ideal_bar, u=sel_box.u)
            i += 1
        self.timer.append(time.time() - start_time)
