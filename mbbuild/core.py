import re
import os
import numpy as np
import time
import itertools

from .util import tostderr
from .shell import *


DELIM = [
    '.',
    '-',
    '_',
    '+'
]


def increment_delimiter(d='.'):
    ix = DELIM.index(d)
    if ix == len(DELIM) - 1:
        out = 0
    else:
        out = ix + 1
    return DELIM[out]


def decrement_delimiter(d='.'):
    ix = DELIM.index(d)
    if ix == 0:
        out = len(DELIM) - 1
    else:
        out = ix - 1
    return DELIM[out]


class Arg(object):
    def __init__(
            self,
            key,
            dtype=str,
            positional=False,
            default=None
    ):
        self.key = key
        self.dtype = dtype
        self.positional = positional
        self.default = default

    def read(self, s):
        if not self.positional:
            if not s.startswith(self.key):
                return None
            s = s[len(self.key):]
        return self.dtype(s)

    def syntax_str(self):
        if self.positional:
            return self.key.upper()
        return self.key + 'VAL'







class Graph(object):
    def __init__(self):
        self.nodes = {}

    def __iter__(self):
        return self.nodes.__iter__()

    def __setitem__(self, key, value):
        self.nodes[key] = value

    def __getitem__(self, key):
        return self.nodes.get(key, None)

    def add(self, node):
        assert not (type(node), node.path) in self.nodes, 'Attempted to re-insert and existing key: %s' % str(type(node), node.path)
        k = (type(node), node.path)
        v = node
        self[k] = v
        node.graph = self

    def compute_paths(self, node):
        out = None
        if node is not None:
            prereq_sets = set()
            for alt in node.prereqs_all:
                if len(alt) == 0:
                    return {(node, None)}
                deps = []
                for dep in alt:
                    paths = self.compute_paths(dep)
                    deps.append(paths)
                if len(deps) > 1:
                    deps = set(itertools.product(*deps))
                else:
                    deps = {(x,) for x in deps[0]}
                deps = {(node, x) for x in deps}

                prereq_sets = prereq_sets.union(deps)
            out = prereq_sets
        return out

    def pretty_print_path(self, path, indent=0):
        out = '-' * (indent + 1) + '  ' + type(path[0]).__name__ + ':  ' +  path[0].path + '\n'
        if path[1] is not None:
            for p in path[1]:
                out += self.pretty_print_path(p, indent=indent + 1)

        return out

    def pretty_print_paths(self, paths=None, node=None):
        assert not (paths is None and node is None), 'pretty_print_paths requires at least 1 non-null argument (paths or node)'
        if paths is None:
            paths = self.compute_paths(node)
        out = ''
        for i, p in enumerate(paths):
            out += str(i+1) + ')\n'
            out += self.pretty_print_path(p)
            out += '\n'

        return out


class MBType(object):
    SUFFIX = ''
    MANIP = ''
    PREREQ_TYPES = []
    ARGS = []

    ARG_OUTER_DELIM = '.'
    ARG_INNER_DELIM = '-'

    REPEATABLE_PREREQ = False
    DESCR_SHORT = 'data'
    DESCR_LONG = (
        "Abstract base class for ModelBlocks types.\n"
    )

    def __init__(
            self,
            name,
            dump=False,
            delim=None
    ):
        if name.endswith(self.SUFFIX):
            if self.SUFFIX != '':
                name = name[:-len(self.SUFFIX)]
        self.directory = os.path.dirname(name)
        if self.directory == '':
            self.directory = '.'
        self.directory += '/'
        self.basename = os.path.basename(name)
        self.prereqs_all = None
        self.prereqs = None
        self.dependencies = None
        self.path = self.directory + self.basename + self.SUFFIX
        self.value = None
        self.max_timestamp_src = None

        self.dump = dump or os.path.exists(self.path)
        self.delim_src = delim

        self.graph = None
        self.started = False

    @property
    def delim(self):
        if self.delim_src is not None:
            return self.DEFAULT_DELIM
        return self.delim_src

    @property
    def timestamp(self):
        if os.path.exists(self.path):
            out = os.path.getmtime(self.path)
        elif self.content is None:
            out = -np.inf
        else:
            out = time.time()

        return out

    @property
    def max_timestamp(self):
        if self.max_timestamp_src == None:
            max_timestamp = self.timestamp
            for s in self.prereqs:
                max_timestamp = max(max_timestamp, s.max_timestamp)
            self.max_timestamp_src = max_timestamp

        return self.max_timestamp_src

    @property
    def content(self):
        if self.exists():
            out = Content(
                data=self.value,
                path=self.path
            )
        else:
            out = None

        return out

    @property
    def output_buffer(self):
        if self.dump:
            out = self.path
        else:
            out = None

        return out

    @classmethod
    def assemble(cls, match):
        return ''.join(match) + cls.SUFFIX

    @classmethod
    def inheritors(cls):
        out = set()
        for c in cls.__subclasses__():
            out.add(c)
            out = out.union(c.inheritors())

        return out

    @classmethod
    def match(cls, path, delim='.'):
        suffix = cls.MANIP + cls.SUFFIX
        out = path.endswith(suffix)
        if len(cls.PREREQ_TYPES) == 0:
            out &= os.path.basename(path[:-len(suffix)]) == ''
        elif len(cls.PREREQ_TYPES) > 1 or cls.REPEATABLE_PREREQ:
            basenames = path[:-len(suffix)].split(delim)
            prereq_types = cls.PREREQ_TYPES[:]
            if cls.REPEATABLE_PREREQ:
                while len(prereq_types) < len(basenames):
                    prereq_types.insert(0, prereq_types[0])

            out &= len(prereq_types) == len(basenames)

        return out

    @classmethod
    def strip_suffix(cls, path):
        suffix = cls.MANIP + cls.SUFFIX
        if suffix != '':
            name_new = path[:-len(suffix)]
        else:
            name_new = path
        return name_new

    @classmethod
    def prereq_parser(cls, i=0):
        if len(cls.ARGS) == 0:
            def parser(path):
                return cls.strip_suffix(path),
        else:
            def parser(path):
                basename = cls.strip_suffix(path)
                basename_split = basename.split(cls.ARG_OUTER_DELIM)
                basename = cls.ARG_OUTER_DELIM.join(basename_split[:-1])
                out = [basename]
                argstr = basename_split[-1]
                argstr = argstr.split(cls.ARG_INNER_DELIM)
                args = [a for a in cls.ARGS if a.positional]
                kwargs = [a for a in cls.ARGS if not a.positional]
                assert len(args) <= len(argstr), 'Expected %d positional arguments, saw %d.' % (len(args), len(argstr))
                for arg in args:
                    s = argstr.pop(0)
                    v = arg.read(s)
                    k = arg.key
                    out.append((k, v))
                for s in argstr:
                    out_cur = None
                    for i in range(len(kwargs)):
                        kwarg = kwargs[i]
                        r = kwarg.read(s)
                        if r is not None:
                            out_cur = (kwarg.key, r)
                            kwargs.pop(i)
                            break

                    assert out_cur is not None, 'Unrecognized keyword argument %d' % s
                    out.append(out_cur)

                for kwarg in kwargs:
                    out.append((kwarg.key, kwarg.default))

                out = tuple(out)
                return out

        return parser

    @classmethod
    def parse_path(cls, path, delim='.'):
        out = None
        if cls.match(path):
            if len(cls.PREREQ_TYPES) > 1 or cls.REPEATABLE_PREREQ:
                basename = cls.strip_suffix(path)
                directory = os.path.dirname(basename)
                if directory == '':
                    directory = '.'
                directory += '/'
                basename = os.path.basename(basename)
                basenames = [directory + x for x in basename.split(delim)]
                prereq_types = cls.PREREQ_TYPES[:]
                if cls.REPEATABLE_PREREQ:
                    while len(prereq_types) < len(basenames):
                        prereq_types.insert(0, prereq_types[0])
                out = []
                for i, (b, p) in enumerate(zip(basenames, prereq_types)):
                    b = b.replace(increment_delimiter(delim), delim)
                    name = b + p.SUFFIX
                    out.append(p.prereq_parser(i=i)(name))
                out = tuple(out)
            else:
                name = path
                out = (cls.prereq_parser()(name),)

        return out

    @classmethod
    def compute_graph(cls, name, graph=None):
        out = None

        if graph is None:
            graph = Graph()
        parsed = cls.parse_path(name)
        if parsed is not None:
            if len(cls.PREREQ_TYPES) == 0:
                name_new = parsed[0][0]
                if os.path.basename(name_new) == '':
                    return {tuple()}
            prereq_types = cls.PREREQ_TYPES[:]
            if cls.REPEATABLE_PREREQ:
                while len(prereq_types) < len(parsed):
                    prereq_types.insert(0, prereq_types[0])
            prereq_set = []
            for (P, p) in zip(prereq_types, parsed):
                dep = set()
                inheritors = P.inheritors().union({P})
                name_new = p[0]
                for c in inheritors:
                    path = name_new + c.SUFFIX
                    prereqs = c.compute_graph(path, graph=graph)
                    if prereqs is not None:
                        p = graph[(c, path)]
                        if p is None:
                            p = c(name_new)
                            graph.add(p)
                            p.add_prereqs(prereqs)
                        dep.add(p)
                if len(dep) == 0:
                    return None
                prereq_set.append(dep)
            if len(prereq_set) > 1:
                prereq_set = itertools.product(*prereq_set)
            else:
                prereq_set = tuple((x,) for x in prereq_set[0])
            prereq_set = set(prereq_set)

            out = prereq_set

        return out

    @classmethod
    def report_api(cls):
        out = '-' * 50 + '\n'
        out += 'Class:                %s\n' % cls.__name__
        out += 'Short description:    %s\n' % cls.DESCR_SHORT
        out += 'Detailed description: %s\n' % cls.DESCR_LONG
        out += 'Prerequisites'
        if cls.REPEATABLE_PREREQ:
            out += ' (repeatable)'
        out += ':\n'
        for x in cls.PREREQ_TYPES:
            out += '  %s\n' % x.__name__
        out += '\n'
        out += 'Syntax: ' + cls.syntax_str()
        out += '\n\n'

        return out

    @classmethod
    def syntax_str(cls):
        out = []
        for i, x in enumerate(cls.PREREQ_TYPES):
            if i == 0 and cls.REPEATABLE_PREREQ:
                s = '<%s>.(<%s>)+' % (x.__name__, x.__name__)
            else:
                s = '<%s>' % x.__name__
            out.append(s)
        out = '.'.join(out)
        if len(cls.ARGS) > 0:
            out += cls.ARG_OUTER_DELIM
            arg_str = ['<%s>' % a.syntax_str() for a in cls.ARGS]
            out += cls.ARG_INNER_DELIM.join(arg_str)
        out += cls.MANIP
        out += cls.SUFFIX

        return out

    def add_prereqs(self, prereqs=None):
        if prereqs is None:
            if self.prereqs is None:
                self.prereqs_all = self.compute_graph(self.path, graph=self.graph)
        else:
            self.prereqs_all = prereqs
        self.prereqs = self.prereqs_all.pop()
        self.prereqs_all.add(self.prereqs)

    def set_value(self, val):
        self.value = val

    def read(self):
        if self.exists():
            with open(self.path, 'r') as f:
                return f.readlines()

    def exists(self):
        return self.value is not None or os.path.exists(self.path)

    def run(self, dry_run=False, force=False):
        prereqs = self.prereqs
        if not self.started:
            self.started = True

            build = force
            build |= self.max_timestamp > self.timestamp
            build |= self.content is None

            out = None
            if build:
                for s in prereqs:
                    s.run(dry_run=dry_run, force=force)

                cmd = self.build(dry_run=dry_run)
                if not isinstance(cmd, list):
                    cmd = [cmd]
                if len(cmd) > 0 and cmd[-1].stdout_src is None and not isinstance(cmd[-1], Dump):
                    cmd[-1].set_stdout(self.path)

                out = None
                for c in cmd:
                    out = c.run(dry_run=dry_run)

                if out is not None:
                    self.set_value(out.data)

            else:
                outstr = '%s is up to date' % self.path
                if not self.dump:
                    outstr += ' (stored in memory)'
                outstr += '.\n'
                tostderr(outstr)

            return out

    def build(self, dry_run=False):
        raise NotImplementedError


class Target(MBType):
    PREREQ_TYPES = [MBType]
    DESCR_LONG = (
        "Special class for executing targets.\n"
    )

    def __init__(
            self,
            names,
            name='Target'
    ):
        super(Target, self).__init__(name, dump=True)
        self.path = names

    @classmethod
    def compute_graph(cls, name, graph=None):
        if graph is None:
            graph = Graph()
        if isinstance(name, str):
            names = name.split()
        else:
            names = name
        out = None
        for name in names:
            parsed = cls.parse_path(name)

            prereq_set = []
            for p in parsed:
                dep = set()
                inheritors = MBType.inheritors() - {Target}
                path = p[0]
                for c in inheritors:
                    prereqs = c.compute_graph(path, graph=graph)
                    if prereqs is not None:
                        p = graph[(c, path)]
                        if p is None:
                            p = c(path, dump=True)
                            graph.add(p)
                            p.add_prereqs(prereqs)
                        else:
                            p.dump = True
                        dep.add(p)
                if len(dep) == 0:
                    return None
                prereq_set.append(dep)
            if len(prereq_set) > 1:
                prereq_set = itertools.product(*prereq_set)
            else:
                prereq_set = tuple(prereq_set[0])
            prereq_set = set(prereq_set)

            if out is None:
                out = [prereq_set]
            else:
                out.append(prereq_set)

        if len(out) > 1:
            out = set(itertools.product(*out))
        else:
            out = {(x,) for x in out[0]}

        return out

    def build(self, dry_run=False):
        return []

    def add_prereqs(self, prereqs=None):
        super(Target, self).add_prereqs(prereqs=prereqs)

        self.graph = self.prereqs[0].graph

    def run(self, dry_run=False, force=False):
        prereqs = self.prereqs
        for x in prereqs:
            x.run(dry_run=dry_run, force=force)