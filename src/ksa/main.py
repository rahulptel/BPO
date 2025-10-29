import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path

from ksa import EpsilonSearch
from moop import factory as moop_factory


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--n_worker', type=int, default=1)
    parser.add_argument('-p', type=str, default='KP')
    args = parser.parse_args()
    return args


def worker_ksa(args, filename, save_path):
    problem = moop_factory(args.p, filename)
    es = EpsilonSearch(problem, save_path)
    es.run()
    es.log()


def main(args):
    root = Path(__file__).parent / 'data' / args.p
    save_path = Path(__file__).parent / 'result' / args.p

    if args.n_worker > 1:
        pool = mp.Pool(args.n_worker)

        results = []
        for filepath in root.iterdir():
            results.append(pool.apply_async(worker_ksa, args=(args, filepath, save_path,)))
        for r in results:
            r.get()

    else:
        # for filepath in root.iterdir():
        #     worker_ksa(args, root / filepath)
        #     break
        worker_ksa(args, root / 'KP_p-3_n-10_ins-1.dat', save_path)


if __name__ == '__main__':
    main(get_args())
