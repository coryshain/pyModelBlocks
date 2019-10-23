import sys
import argparse

from .core import MBType
from .linetrees import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Run one or more MB targets
    ''')
    argparser.add_argument('targets', nargs='+', help='Space-delimited list of target strings')
    argparser.add_argument('-n', '--dry_run', action='store_true', help='Log commands that would be executed to stderr, but do not run them.')
    argparser.add_argument('-p', '--paths', action='store_true', help='Log available dependency paths for the target to stderr, rather than building it.')
    argparser.add_argument('-H', '--api_help', action='store_true', help='Print API strings for available classes.')
    args = argparser.parse_args()

    if args.api_help:
        for cls in MBType.inheritors():
            sys.stderr.write(cls.report_api())
        exit()

    x = Target(args.targets)
    x.add_prereqs()
    if args.paths:
        sys.stderr.write(x.graph.pretty_print_paths(node=x.prereqs[0]))
        exit()

    x.run(dry_run=args.dry_run)
