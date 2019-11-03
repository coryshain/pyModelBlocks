from .core import *
from . import tree


class LineTrees(MBType):
    SUFFIX = '.linetrees'
    DESCR_SHORT =  'linetrees'
    DESCR_LONG = (
        "Abstract base class for linetrees types.\n"
    )
    
    @classmethod
    def abstract(cls):
        return cls.__name__ == 'LineTrees'


class LineTreesUpper(LineTrees):
    MANIP = '.upper'
    PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'uppercased linetrees'
    DESCR_LONG = (
        "Uppercase words in trees.\n"
    )

    def body(self):
        def out(inputs):
            t = tree.Tree()
            outputs = []
            for x in inputs:
                t.read(x)
                t.upper()
                outputs.append(str(t))

            return outputs

        return out


class LineTreesLower(LineTrees):
    MANIP = '.lower'
    PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'lowercased linetrees'
    DESCR_LONG = (
        "Lowercase words in trees.\n"
    )
    
    def body(self):
        def out(inputs):
            t = tree.Tree()
            outputs = []
            for x in inputs:
                t.read(x)
                t.lower()
                outputs.append(str(t))
                
            return outputs

        return out


class LineTreesFirst(LineTrees):
    MANIP = 'first'
    PREREQ_TYPES = [LineTrees]
    ARG_TYPES = [Arg('n', dtype=int, positional=True)]
    DESCR_SHORT = 'first N linetrees'
    DESCR_LONG = (
        "Truncate linetrees file to contain the first N lines.\n"
    )

    def body(self):
        def out(inputs, n):
            return inputs[:n]

        return out


class LineTreesLast(LineTrees):
    MANIP = 'last'
    PREREQ_TYPES = [LineTrees]
    ARG_TYPES = [Arg('n', dtype=int, positional=True)]
    DESCR_SHORT = 'last N linetrees'
    DESCR_LONG = (
        "Truncate linetrees file to contain the last N lines.\n"
    )

    def body(self):
        def out(inputs, n):
            return inputs[-n:]

        return out


class LineTreesMerged(LineTrees):
    MANIP = '.merged'
    PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'merged linetrees'
    REPEATABLE_PREREQ = True
    DESCR_LONG = (
        "Concatenate linetrees files.\n"
    )

    def body(self):
        def out(*args):
            outputs = []
            for x in args:
                outputs += x
            return outputs
        return out


class LineTreesMergedPrefix(LineTreesMerged):
    HAS_PREFIX = True
