import argparse

from . import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Display documentation for available MB target types
    ''')
    args = argparser.parse_args()

    for cls in MBType.inheritors():
        sys.stdout.write(cls.report_api())