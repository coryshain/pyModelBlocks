import argparse

from . import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Run one or more MB targets
    ''')
    argparser.add_argument('targets', nargs='+', help='Space-delimited list of target strings')
    argparser.add_argument('-C', '--clear_history', action='store_true', help='Clear the history of pointers to external files. Subsequently, all dependencies to external files will be treated as old.')
    argparser.add_argument('-n', '--dry_run', action='store_true', help='Log commands that would be executed to stderr, but do not run them.')
    argparser.add_argument('-j', '--concurrent', default=None, const=0, type=int, nargs='?', action='store', help='Run in concurrent mode. Optional INT limit on number of concurrent processes.')
    argparser.add_argument('-p', '--paths', action='store_true', help='Log available dependency paths for the target to stderr, rather than building it.')
    args = argparser.parse_args()

    if args.clear_history:
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)

    concurrent = args.concurrent is not None
    n_concurrent = None if args.concurrent in [None, 0] else args.concurrent
    if concurrent:
        import ray
        ray.init(num_cpus=n_concurrent)
    else:
        ray = None

    g = Graph(args.targets, process_scheduler=ray)
    if args.paths:
        sys.stderr.write(g.pretty_print_paths())
        exit()

    t0 = time.time()
    out = g.get(dry_run=args.dry_run)
    t1 = time.time()

    print('\nBuild time: %.1fs' % (t1 - t0))
