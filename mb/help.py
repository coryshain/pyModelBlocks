import argparse

from . import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Display documentation.
    If run without flags, displays text API for all ModelBlocks types.
    ''')
    argparser.add_argument('-c', '--config', nargs='+', help='Display current values for each of list of config params.')
    args = argparser.parse_args()

    if args.config:
        for key in args.config:
            val = USER_SETTINGS.get(key, DEFAULT_SETTINGS.get(key, None))
            if val is None:
                print('No value for key "%s"' % key)
            else:
                print(val)
    else:
        for cls in MBType.inheritors():
            sys.stdout.write(cls.report_api())
            sys.stdout.flush()