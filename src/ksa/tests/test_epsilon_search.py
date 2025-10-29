from pathlib import Path

import numpy as np

from ksa import EpsilonSearch
from moop import factory as moop_factory


def test_is_new_nondominated_solution():
    root = Path(__file__).parent.parent / 'data' / 'KP'
    problem = moop_factory('knapsack', root / 'KP_p-3_n-10_ins-1.dat')
    es = EpsilonSearch(problem)

    es.moop.n_objectives = 2
    # populate dummy Z_n
    es.Z_n.append([-20, 20])
    es.Z_n.append([-10, 10])
    es.Z_n.append([20, -20])

    # check z in Z_n
    assert es._is_new_nondominated_solution([10, 0])
    assert es._is_new_nondominated_solution([-15, 12])
    assert es._is_new_nondominated_solution([15, 1])

    assert not es._is_new_nondominated_solution([-20, 20])
    assert not es._is_new_nondominated_solution([-10, 10])
    assert not es._is_new_nondominated_solution([20, -20])


def test_update_list():
    from ksa.utils import Box
    root = Path(__file__).parent.parent / 'data' / 'KP'
    problem = moop_factory('knapsack', root / 'KP_p-3_n-10_ins-1.dat')
    es = EpsilonSearch(problem)

    Box.z_ideal_bar = np.asarray([-100, -100])
    b = Box(l=np.asarray([-100, -100]), u=np.asarray([100, 100]))
    es.L[b.id] = b

    es._update_list(np.asarray([0, 0]))
    assert len(es.L.keys()) == 4
    boxes = {"-100_-100": np.asarray([0, 0]),
             "-100_0": np.asarray([0, 100]),
             "0_-100": np.asarray([100, 0]),
             "0_0": np.asarray([100, 100])}
    for b in es.L.values():
        key = "_".join(map(str, map(int, b.l)))
        print(key, b.u)
        assert key in boxes
        assert (boxes[key] == b.u).all()

    es._update_list(np.asarray([-50, -50]))
    assert len(es.L.keys()) == 9
    boxes = {"-100_-100": np.asarray([-50, -50]),
             "-100_-50": np.asarray([-50, 0]),
             "-50_-100": np.asarray([0, -50]),
             "-50_-50": np.asarray([0, 0]),
             "-100_0": np.asarray([-50, 100]),
             "-50_0": np.asarray([0, 100]),
             "0_-100": np.asarray([100, -50]),
             "0_-50": np.asarray([100, 0]),
             "0_0": np.asarray([100, 100])}
    for b in es.L.values():
        key = "_".join(map(str, map(int, b.l)))
        assert key in boxes
        assert (boxes[key] == b.u).all()

    es._update_list(np.asarray([50, 50]))
    assert len(es.L.keys()) == 16
    boxes = {"-100_-100": np.asarray([-50, -50]),
             "-100_-50": np.asarray([-50, 0]),
             "-50_-100": np.asarray([0, -50]),
             "-50_-50": np.asarray([0, 0]),
             "-100_0": np.asarray([-50, 50]),
             "-100_50": np.asarray([-50, 100]),
             "-50_0": np.asarray([0, 50]),
             "-50_50": np.asarray([0, 100]),
             "0_-100": np.asarray([50, -50]),
             "0_-50": np.asarray([50, 0]),
             "50_-100": np.asarray([100, -50]),
             "50_-50": np.asarray([100, 0]),
             "0_0": np.asarray([50, 50]),
             "0_50": np.asarray([50, 100]),
             "50_0": np.asarray([100, 50]),
             "50_50": np.asarray([100, 100])}
    for b in es.L.values():
        key = "_".join(map(str, map(int, b.l)))
        assert key in boxes
        assert (boxes[key] == b.u).all()


def test_remove_rectangle():
    from ksa.utils import Box
    root = Path(__file__).parent.parent / 'data' / 'KP'
    problem = moop_factory('knapsack', root / 'KP_p-3_n-10_ins-1.dat')
    es = EpsilonSearch(problem)

    Box.z_ideal_bar = np.asarray([-100, -100])
    b = Box(l=np.asarray([-100, -100]), u=np.asarray([100, 100]))
    es.L[b.id] = b

    es._update_list(np.asarray([0, 0]))
    es._remove_rectangle(l=np.asarray([0, 0]), u=np.asarray([100, 100]))
    assert len(es.L.keys()) == 3

    es._update_list(np.asarray([-50, -50]))
    es._remove_rectangle(l=np.asarray([-50, -50]), u=np.asarray([0, 0]))
    assert len(es.L.keys()) == 7

    es._remove_rectangle(l=np.asarray([-100, -100]), u=np.asarray([-50, 100]))
    assert len(es.L.keys()) == 4

    es._update_list(np.asarray([50, -25]))
    es._remove_rectangle(l=np.asarray([50, -25]), u=np.asarray([100, 0]))
    assert len(es.L.keys()) == 7

    boxes = {"-50_0": np.asarray([0, 100]),
             "-50_-100": np.asarray([0, -50]),
             "0_-25": np.asarray([50, 0]),
             "0_-50": np.asarray([50, -25]),
             "0_-100": np.asarray([50, -50]),
             "50_-100": np.asarray([100, -50]),
             "50_-50": np.asarray([100, -25])}
    for b in es.L.values():
        key = "_".join(map(str, map(int, b.l)))
        assert key in boxes
        assert (boxes[key] == b.u).all()


# def test_run():
#     pass
