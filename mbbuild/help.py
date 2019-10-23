import argparse

from .core import MBType
from .linetrees import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Display documentation for available MB target types
    ''')
    args = argparser.parse_args()

    for cls in MBType.inheritors():
        sys.stderr.write(cls.report_api())