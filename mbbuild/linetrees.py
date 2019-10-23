import re
import os
import numpy as np

from .util import tostderr
from .shell import *
from .core import *


class LineTrees(MBType):
    SUFFIX = '.linetrees'
    DESCR_SHORT =  'linetrees'
    DESCR_LONG = (
        "Abstract base class for linetrees types.\n"
    )

    def __init__(
            self,
            name,
            dump=False
    ):
        super(LineTrees, self).__init__(name, dump=dump)


class LineTreesNatstor(LineTrees):
    MANIP = 'naturalstories'
    DESCR_SHORT = 'naturalstories gold linetrees'
    DESCR_LONG = (
        "Source trees (hand-annotated) for the Natural Stories corpus.\n"
    )

    def __init__(
            self,
            name,
            dump=False
    ):
        super(LineTreesNatstor, self).__init__(name, dump=dump)

    def build(self, dry_run=False):
        outstr = '(X (X This) (X (X is) (X (X a) (X (X test)))))\n' * 100

        cmd = [
            ShellCommand('mkdir', ['-p', self.directory]),
            ShellCommand('echo', [outstr], stdout=self.output_buffer)
        ]

        return cmd


class LineTreesNatstor2(LineTrees):
    MANIP = 'naturalstories'
    DESCR_SHORT = 'naturalstories gold linetrees 2'
    DESCR_LONG = (
        "Alternative annotation for the Natural Stories corpus.\n"
    )

    def __init__(
            self,
            name,
            dump=False
    ):
        super(LineTreesNatstor2, self).__init__(name, dump=dump)

    def build(self, dry_run=False):
        outstr = '(X (X This) (X (X is) (X (X a) (X (X test)))))\n' * 100

        cmd = [
            ShellCommand('mkdir', ['-p', self.directory]),
            ShellCommand('echo', [outstr], stdout=self.output_buffer)
        ]

        return cmd


class LineTreesUpper(LineTrees):
    MANIP = '.upper'
    PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'uppercased linetrees'
    DESCR_LONG = (
        "Convert any text in a linetrees file to uppercase.\n"
    )

    def __init__(
            self,
            name,
            dump=False
    ):
        super(LineTreesUpper, self).__init__(name, dump=dump)

    def build(self, dry_run=False):
        assert len(self.prereqs) == 1, 'Expected exactly 1 input to %s, got %d' % (type(self).__name__, len(self.prereqs))

        if dry_run:
            outputs = None
        else:
            inputs = self.prereqs[0].content.data
            outputs = [x.upper() for x in inputs]

        return Dump(outputs, stdout=self.output_buffer, descr=self.DESCR_SHORT)


class LineTreesLower(LineTrees):
    MANIP = '.lower'
    PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'lowercased linetrees'
    DESCR_LONG = (
        "Convert any text in a linetrees file to lowercase.\n"
    )

    def __init__(
            self,
            name,
            dump=False
    ):
        super(LineTreesLower, self).__init__(name, dump=dump)

    def build(self, dry_run=False):
        assert len(self.prereqs) == 1, 'Expected exactly 1 input to %s, got %d' % (type(self).__name__, len(self.prereqs))

        if dry_run:
            outputs = None
        else:
            inputs = self.prereqs[0].content.data
            outputs = [x.lower() for x in inputs]

        return Dump(outputs, stdout=self.output_buffer, descr=self.DESCR_SHORT)


class LineTreesFirst(LineTrees):
    MANIP = 'first'
    PREREQ_TYPES = [LineTrees]
    ARGS = [Arg('n', dtype=int, positional=True)]
    DESCR_SHORT = 'first N linetrees'
    DESCR_LONG = (
        "Truncate linetrees file to contain the first N lines.\n"
    )

    def __init__(
            self,
            name,
            dump=False
    ):
        super(LineTreesFirst, self).__init__(name, dump=dump)

    def build(self, dry_run=False):
        assert len(self.prereqs) == 1, 'Expected exactly 1 input to %s, got %d' % (type(self).__name__, len(self.prereqs))

        parsed = self.parse_path(self.path)[0]
        n = parsed[1][1]

        if dry_run:
            outputs = None
        else:
            outputs = self.prereqs[0].content.data[:n]

        return Dump(outputs, stdout=self.output_buffer, descr=self.DESCR_SHORT)


class LineTreesLast(LineTrees):
    MANIP = 'last'
    PREREQ_TYPES = [LineTrees]
    ARGS = [Arg('n', dtype=int, positional=True)]
    DESCR_SHORT = 'last N linetrees'
    DESCR_LONG = (
        "Truncate linetrees file to contain the last N lines.\n"
    )

    def __init__(
            self,
            name,
            dump=False
    ):
        super(LineTreesLast, self).__init__(name, dump=dump)

    def build(self, dry_run=False):
        assert len(self.prereqs) == 1, 'Expected exactly 1 input to %s, got %d' % (type(self).__name__, len(self.prereqs))

        parsed = self.parse_path(self.path)[0]
        n = parsed[1][1]

        if dry_run:
            outputs = None
        else:
            inputs = self.prereqs[0].content.data[-n:]
            outputs = inputs

        return Dump(outputs, stdout=self.output_buffer, descr=self.DESCR_SHORT)


class LineTreesMerged(LineTrees):
    MANIP = '.merged'
    PREREQ_TYPES = [LineTrees]
    DESCR_SHORT = 'merged linetrees'
    REPEATABLE_PREREQ = True
    DESCR_LONG = (
        "Concatenate linetrees files.\n"
    )

    def __init__(
            self,
            name,
            dump=False
    ):
        super(LineTreesMerged, self).__init__(name, dump=dump)

    def build(self, dry_run=False):
        if dry_run:
            outputs = None
        else:
            inputs = self.prereqs
            outputs = []
            for x in inputs:
                outputs += x.content.data

        return Dump(outputs, stdout=self.output_buffer, descr=self.DESCR_SHORT)