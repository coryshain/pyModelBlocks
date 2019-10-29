import sys
import shutil
import numpy as np
import time
import itertools
import re
import multiprocessing

from .cmd import *

if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_RESOURCES_DIR = os.path.join(ROOT_DIR, 'static_resources')
DEFAULT_PATH = os.path.join(ROOT_DIR, '.defaults.ini')
CONFIG_PATH = os.path.join(ROOT_DIR, '.config.ini')
HISTORY_PATH = os.path.join(ROOT_DIR, '.hist.ini')

DEFAULT_CONFIG = configparser.ConfigParser()
DEFAULT_CONFIG.optionxform = str
DEFAULT_CONFIG.read(DEFAULT_PATH)
DEFAULTS = DEFAULT_CONFIG['settings']


DELIM = [
    '.',
    '-',
    '_',
    '+'
]
NON_VAR_CHARS = re.compile('\W')


def normalize_class_name(name):
    out = NON_VAR_CHARS.sub('', name)
    if len(out) > 0:
        out = out[0].upper() + out[1:]
    return out


def create_classes_from_dir(directory, parent_name=''):
    out = []
    for static_file in os.listdir(directory):
        if not static_file == '__init__.py':
            parent_name = normalize_class_name(parent_name)
            path = os.path.join(directory, static_file)
            static_file_parts = static_file.split('.')
            if len(static_file_parts) > 1:
                descr = ''.join(static_file_parts[:-1])
            else:
                descr = static_file_parts[0]
            class_name = normalize_class_name(descr)

            if os.path.isdir(path):
                parent_name_cur = parent_name + class_name
                out += create_classes_from_dir(path, parent_name=parent_name_cur)
            else:
                class_name = parent_name + class_name

                attr_dict = {
                    'DEFAULT_LOCATION': path,
                    'DESCR_SHORT': descr,
                    'DESCR_LONG': descr + ' utility'
                }

                out.append(type(class_name, (StaticResource,), attr_dict))

    return out


def get_timestamp(path):
    if os.path.exists(path):
        t = os.path.getmtime(path)
        if os.path.isdir(path):
            for c in os.listdir(path):
                t = max(t, get_timestamp(os.path.join(path, c)))
    else:
        t = -np.inf

    return t


def increment_delimiters(s):
    """
    Increase the depth of all delimiters in **s**.
    
    :param ``s``: str; the input string
    :return: ``str``; **s** with deeper delimiters (shifted towards the end of DELIM)
    """
    
    out = ''
    for c in s:
        try:
            i = DELIM.index(c)
            assert i < len(DELIM), 'Cannot increment delimiters for "%s" because it already contains the deepest delimiter "%s". Your target may involve too much nesting.' % (s, DELIM[-1])
            out += DELIM[i+1]
        except ValueError:
            out += c

    return out


def decrement_delimiters(s):
    """
    Decrease the depth of all delimiters in **s**.
    
    :param s: ``str``; the input string
    :return: ``str``; **s** with shallower delimiters (shifted towards the beginning of DELIM)
    """
    
    out = ''
    for c in s:
        try:
            i = DELIM.index(c)
            assert i < len(DELIM), 'Cannot decrement delimiters for "%s" because it already contains the shallowest delimiter "%s". Your target may involve too much nesting.' % (s, DELIM[0])
            out += DELIM[i-1]
        except ValueError:
            out += c

    return out


class Arg(object):
    """
    Object representing an argument to a data transform (positional or keyword).
    Arguments parameterize all aspects of the transform except the input data,
    which should be computed as prerequisites in the dependency graph.
    Positional arguments are treated as obligatory, keyword arguments are treated as optional.

    :param key: ``str``; name of the argument
    :param dtype: ``type``; data type of the argument
    :param positional: ``bool``; whether the argument is positional
    :param default: default value for the argument
    """

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
    def __init__(self, target_paths=None, processes=None):
        self.target_paths = None
        self.targets_all_paths = None
        self.targets = None
        self.processes = processes
        self.nodes = {}

        self.add_targets(target_paths)

    def __iter__(self):
        return self.nodes.__iter__()

    def __setitem__(self, key, value):
        self.nodes[key] = value

    def __getitem__(self, key):
        return self.nodes.get(key, None)

    def add_targets(self, targets):
        if self.target_paths is None:
            self.target_paths = []
        if targets is not None:
            if isinstance(targets, str):
                targets = targets.split()
            else:
                targets = list(targets)

            targets = [os.path.normpath(t) for t in targets]

            targets += self.target_paths
            targets = sorted(list(set(targets)))
            self.target_paths = targets

            self.construct_graph()

    def add_node(self, node):
        assert not (type(node), node.path) in self.nodes, 'Attempted to re-insert and existing key: %s' % str(type(node), node.path)
        k = (type(node), node.path)
        v = node
        self[k] = v
        node.graph = self

    def construct_graph(self):
        # Compute all dependency paths to target set
        targets_all_paths = []
        for target_path in self.target_paths:
            target = set()
            inheritors = MBType.inheritors()
            for c in inheritors:
                target_prereqs_cur, target_static_prereqs_cur = self.construct_subgraph(c, target_path)
                if target_prereqs_cur is not None:
                    p = self[(c, target_path)]
                    if p is None:
                        if issubclass(c, StaticResource):
                            p = c()
                        else:
                            p = c(target_path)
                        self.add_node(p)
                        p.add_prereqs(prereqs=target_prereqs_cur, static_prereqs=target_static_prereqs_cur)
                    p.dump = True
                    p.intermediate = False
                    target.add(p)
            targets_all_paths.append(target)

        # Filter out failed dependency paths, report problems if none succeed
        exists = [True if len(x) > 0 else False for x in targets_all_paths]
        self.targets_all_paths = targets_all_paths

        null_targs = []
        for i, x in enumerate(exists):
            if not x:
                null_targs.append(self.target_paths[i])

        null_targs_str = ', '.join(null_targs)
        assert len(self.targets_all_paths) > 0, 'Recipes could not be found for some targets: %s' % null_targs_str

        self.targets = []
        for t in self.targets_all_paths:
            candidates = list(t)
            self.targets.append(candidates[0])

    def construct_subgraph(self, cls, path):
        prereqs_all_paths = None
        static_prereqs = None

        parsed = cls.parse_path(path)
        if parsed is not None:
            # STATIC PREREQS
            static_prereq_types = cls.static_prereq_types()[:]
            static_prereqs = []
            for c in static_prereq_types:
                prereq_path = c.infer_paths()[0]
                prereqs_cur, static_prereqs_cur = self.construct_subgraph(c, prereq_path)
                p = self[(c, prereq_path)]
                if p is None:
                    p = c()
                    self.add_node(p)
                    p.add_prereqs(prereqs=prereqs_cur, static_prereqs=static_prereqs_cur)
                static_prereqs.append(p)
            static_prereqs = static_prereqs

            if len(cls.prereq_types(path)) == 0:
                return [], static_prereqs

            # DYNAMIC PREREQS
            prereq_types = cls.prereq_types(path)[:]
            if cls.repeatable_prereq():
                while len(prereq_types) < len(parsed):
                    prereq_types.insert(0, prereq_types[0])
            prereqs_all_paths = []
            for (P, p) in zip(prereq_types, parsed):
                dep = set()
                inheritors = P.inheritors().union({P})
                name_new = p['basename']
                for c in inheritors:
                    prereq_path = name_new + c.suffix()
                    prereqs_cur, static_prereqs_cur = self.construct_subgraph(c, prereq_path)
                    if prereqs_cur is not None:
                        p = self[(c, prereq_path)]
                        if p is None:
                            if issubclass(c, StaticResource):
                                p = c()
                            else:
                                p = c(name_new)
                            self.add_node(p)
                            p.add_prereqs(prereqs=prereqs_cur, static_prereqs=static_prereqs_cur)
                        dep.add(p)
                if len(dep) == 0:
                    return None, None
                prereqs_all_paths.append(dep)

        return prereqs_all_paths, static_prereqs

    def compute_paths(self, node=None):
        prereq_sets = set()
        if node is not None:
            prereqs_all_paths = self.targets_all_paths
        else:
            prereqs_all_paths = node.prereqs_all
        for alt in prereqs_all_paths:
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

    def run(self, dry_run=False, force=False):
        # Make any needed directories
        directories_to_make = set()
        for t in self.targets:
            directories_to_make = directories_to_make.union(t.directories_to_make(force=force))
        directories_to_make = sorted(list(directories_to_make))
        if len(directories_to_make) > 0:
            tostderr('Making directories:\n')
            for d in directories_to_make:
                tostderr('  %s\n' % d)
                if not dry_run:
                    os.makedirs(d)

        # Run targets
        intermediate_paths = []
        for t in self.targets:
            intermediate_paths += t.run(dry_run=dry_run, force=force)

        # Clean up intermediate targets
        if len(intermediate_paths) > 0:
            tostderr('Garbage collecting intermediate files and directories:\n')
            for p in intermediate_paths:
                p_str = '  %s' % p
                if os.path.isdir(p):
                    p_str += ' (entire directory)\n'
                    if dry_run:
                        rm = lambda x: x
                    else:
                        rm = shutil.rmtree
                else:
                    p_str += '\n'
                    if dry_run:
                        rm = lambda x: x
                    else:
                        rm = os.remove
                tostderr(p_str)
                rm(p)


class MBType(object):
    SUFFIX = ''
    MANIP = ''
    PREREQ_TYPES = []
    STATIC_PREREQ_TYPES = []
    REPEATABLE_PREREQ = False

    HAS_PREFIX = False
    HAS_SUFFIX = False
    ARGS = []

    DESCR_SHORT = 'data'
    DESCR_LONG = (
        "Abstract base class for ModelBlocks types.\n"
    )

    def __init__(self, path, out_mode='w'):
        path = os.path.normpath(path)
        if path.endswith(self.suffix()):
            if self.suffix() != '':
                path = path[:-len(self.suffix())]
        self.directory = os.path.dirname(path)
        self.basename = os.path.basename(path)
        self.static_prereqs = None
        self.prereqs_all_paths = None
        self.prereqs = None
        self.dependencies = None
        self.path = os.path.join(self.directory, self.basename + self.suffix())
        self.out_mode = out_mode
        self.data_src = Data(path=self.path, out_mode=out_mode)

        self.dump = os.path.exists(self.path)

        self.graph = None
        self.started = False
        self.ready = False
        self.intermediate = True

    @property
    def timestamp(self):
        return get_timestamp(self.path)

    @property
    def max_timestamp(self):
        max_timestamp = self.timestamp
        for s in self.prereqs + self.static_prereqs:
            max_timestamp = max(max_timestamp, s.max_timestamp)

        return max_timestamp

    @property
    def data(self):
        if self.data_src.data is None:
            self.data_src.set_data(None)
        return self.data_src.data

    @property
    def output_buffer(self):
        if self.dump:
            out = self.path
        else:
            out = None

        return out

    @classmethod
    def suffix(cls):
        return cls.SUFFIX

    @classmethod
    def manip(cls):
        return cls.MANIP

    @classmethod
    def prereq_types(cls, path=None):
        return cls.PREREQ_TYPES

    @classmethod
    def static_prereq_types(cls):
        return cls.STATIC_PREREQ_TYPES

    @classmethod
    def repeatable_prereq(cls):
        return cls.REPEATABLE_PREREQ

    @classmethod
    def has_prefix(cls):
        return cls.HAS_PREFIX

    @classmethod
    def has_suffix(cls):
        return cls.HAS_SUFFIX

    @classmethod
    def args(cls):
        return cls.ARGS

    @classmethod
    def descr_short(cls):
        return cls.DESCR_SHORT

    @classmethod
    def descr_long(cls):
        return cls.DESCR_LONG

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
    def match(cls, path):
        suffix = cls.manip() + cls.suffix()
        out = path.endswith(suffix)
        if len(cls.prereq_types(path)) == 0:
            out &= os.path.basename(path[:-len(suffix)]) == ''
        elif len(cls.prereq_types(path)) > 1 or cls.repeatable_prereq():
            basenames = path[:-len(suffix)].split(DELIM[0])
            if cls.has_prefix():
                basenames = basenames[1:]
            if cls.has_suffix():
                basenames = basenames[:1]
            prereq_types = cls.PREREQ_TYPES[:]
            if cls.repeatable_prereq():
                while len(prereq_types) < len(basenames):
                    prereq_types.insert(0, prereq_types[0])

            out &= len(prereq_types) == len(basenames)

        return out

    @classmethod
    def strip_suffix(cls, path):
        suffix = cls.manip() + cls.suffix()
        if suffix != '':
            name_new = path[:-len(suffix)]
        else:
            name_new = path
        return name_new

    @classmethod
    def path_parser_no_args(cls, path):
        return {'basename': cls.strip_suffix(path)}

    @classmethod
    def path_parser_args(cls, path):
        basename = cls.strip_suffix(path)
        basename_split = basename.split(DELIM[0])
        basename = DELIM[0].join(basename_split[:-1])
        out = {'basename': basename}
        argstr = basename_split[-1]
        argstr = argstr.split(DELIM[1])
        args = [a for a in cls.args() if a.positional]
        kwargs = [a for a in cls.args() if not a.positional]
        assert len(args) <= len(argstr), 'Expected %d positional arguments, saw %d.' % (len(args), len(argstr))
        for arg in args:
            s = argstr.pop(0)
            out[arg.key] = arg.read(s)
        for s in argstr:
            out_cur = None
            for i in range(len(kwargs)):
                kwarg = kwargs[i]
                r = kwarg.read(s)
                if r is not None:
                    out_cur = {kwarg.key: r}
                    kwargs.pop(i)
                    break

            assert out_cur is not None, 'Unrecognized keyword argument %d' % s
            out.update(out_cur)

        for kwarg in kwargs:
            out[kwarg.key] = kwarg.default

        return out

    @classmethod
    def path_parser(cls, path):
        if len(cls.args()) == 0:
            parser = cls.path_parser_no_args(path)
        else:
            parser = cls.path_parser_args(path)

        return parser

    @classmethod
    def parse_path(cls, path):
        out = None
        path = os.path.normpath(path)
        if cls.match(path):
            if len(cls.PREREQ_TYPES) > 1 or cls.repeatable_prereq():
                basename = cls.strip_suffix(path)
                directory = os.path.dirname(basename)
                basename = os.path.basename(basename)
                basenames = basename.split(DELIM[0])
                if cls.has_prefix():
                    prefix = decrement_delimiters(basenames[0])
                    basenames = basenames[1:]
                else:
                    prefix = ''
                if cls.has_suffix():
                    suffix = decrement_delimiters(basenames[-1])
                    basenames = basenames[:1]
                else:
                    suffix = ''
                basenames = [os.path.join(directory, DELIM[0].join([y for y in (prefix, decrement_delimiters(x), suffix) if y != ''])) for x in basenames]
                prereq_types = cls.prereq_types(path)[:]
                if cls.repeatable_prereq():
                    while len(prereq_types) < len(basenames):
                        prereq_types.insert(0, prereq_types[0])
                out = []
                for b, p in zip(basenames, prereq_types):
                    name = b + p.suffix()
                    out.append(p.path_parser(name))
            else:
                name = path
                out = [cls.path_parser(name)]

        return out

    @classmethod
    def report_api(cls):
        out = '-' * 50 + '\n'
        out += 'Class:                %s\n' % cls.__name__
        if issubclass(cls, ExternalResource) and cls.url() is not None:
            out += 'URL:                  %s\n' % cls.url()
        out += 'Short description:    %s\n' % cls.descr_short()
        out += 'Detailed description: %s\n' % cls.descr_long()
        out += 'Prerequisites'
        if cls.repeatable_prereq():
            out += ' (repeatable)'
        out += ':\n'
        for x in cls.prereq_types():
            out += '  %s\n' % x.__name__
        out += '\n'
        out += 'Syntax: ' + cls.syntax_str()
        out += '\n\n'

        return out

    @classmethod
    def syntax_str(cls):
        if issubclass(cls, StaticResource):
            out = cls.infer_paths()[0]
        else:
            out = []
            if cls.has_prefix():
                out.append('<PREFIX>')
            for i, x in enumerate(cls.prereq_types()):
                name = x.__name__
                if i == 0 and cls.repeatable_prereq():
                    s = '<%s>(.<%s>)+' % (name, name)
                else:
                    s = '<%s>' % name
                out.append(s)
            if cls.has_suffix():
                out.append('<SUFFIX>')
            out = '(<DIR>/)' + DELIM[0].join(out)
            if len(cls.args()) > 0:
                out += DELIM[0]
                arg_str = ['<%s>' % a.syntax_str() for a in cls.args()]
                out += DELIM[1].join(arg_str)
            out += cls.manip()
            out += cls.suffix()

        return out
    
    def body(self):
        return None

    def add_prereqs(self, prereqs, static_prereqs=None):
        self.prereqs_all_paths = prereqs
        if static_prereqs is not None:
            self.static_prereqs = static_prereqs
        assert self.prereqs_all_paths is not None, 'No recipe to make %s' % self.path
        prereqs = []
        for p in self.prereqs_all_paths:
            candidates = list(p)
            prereqs.append(candidates[0])

        self.prereqs = prereqs

    def set_data(self, val):
        if isinstance(val, Data):
            self.data_src = val
        else:
            self.data_src.set_data(val)

    def read(self):
        if self.exists():
            with open(self.path, 'r') as f:
                return f.readlines()

    def dump_data(self, path=None):
        if path is None:
            path = self.path
        self.data_src.dump(path)

    def exists(self):
        return self.data is not None

    def directories_to_make(self, force=False):
        out = set()
        build = force or (self.max_timestamp > self.timestamp)
        if build:
            if self.dump and len(self.directory) > 0 and not os.path.exists(self.directory):
                out.add(self.directory)
            for p in self.prereqs:
                out = out.union(p.directories_to_make(force=force))
        return out

    def run(self, dry_run=False, force=False):
        intermediate_paths = []
        if not self.started:
            self.started = True

            build = force or (self.max_timestamp > self.timestamp)

            if build:
                for s in self.static_prereqs:
                    s.run(dry_run=dry_run, force=force)

                for s in self.prereqs:
                    intermediate_paths += s.run(dry_run=dry_run, force=force)

                cmd = self.build(dry_run=dry_run)
                if not isinstance(cmd, list):
                    cmd = [cmd]

                if len(cmd) > 0 and isinstance(cmd[-1], str): # Final command is a shell command string
                    self.dump = True

                data = None
                if not issubclass(self.__class__, StaticResource) or not os.path.exists(self.path):
                    for c in cmd:
                        if self.dump:
                            path = self.path
                        else:
                            path = None
                        if isinstance(c, str):
                            c = ShellCommand(c, path=path)
                        elif hasattr(c, '__call__'):
                            c = PyCommand(c, path=path)
                        else:
                            raise ValueError("Can't construct command for object: %s" % str(c))
                        data = c.run(dry_run=dry_run)
                self.set_data(data)
                if self.dump and not dry_run:
                    self.dump_data()

                if self.intermediate and (force or self.dump):
                    intermediate_paths.append(self.path)

            else:
                outstr = '%s is up to date' % self.path
                if os.path.isfile(self.path) and not self.dump:
                    outstr += ' (stored in memory)'
                outstr += '.\n'
                tostderr(outstr)

        return intermediate_paths

    def build(self, dry_run=False):
        return self.body()


class StaticResource(MBType):
    DEFAULT_LOCATION = ''
    DESCR_LONG = (
        "Abstract base class for dependency to a static resource.\n"
    )

    def __init__(self):
        super(StaticResource, self).__init__(self.DEFAULT_LOCATION)

    @classmethod
    def default_location(cls):
        return cls.DEFAULT_LOCATION

    @classmethod
    def infer_paths(cls):
        path = os.path.normpath(cls.default_location())
        return path, path, path

    @classmethod
    def match(cls, path):
        out = os.path.abspath(path) == os.path.abspath(cls.infer_paths()[0])
        return out

    def build(self, dry_run=False):
        assert os.path.exists(self.path), '%s is a missing static resource and cannot be built. Check to make sure you have the required resources.' % self.path


class ExternalResource(StaticResource):
    URL = None
    PARENT_RESOURCE = None
    DESCR_LONG = (
        "Abstract base class for dependency to external resource.\n"
    )

    def __init__(self):
        super(ExternalResource, self).__init__()

        self.path, self.rel_path_cur, self.rel_path_prev = self.infer_paths()
        self.basename = os.path.basename(self.path)
        self.directory = os.path.dirname(self.path)

    @property
    def timestamp(self):
        paths = [self.path]
        times = []
        if self.parent_resource() is not None:
            paths.append(self.parent_resource().infer_paths()[0])
        for path in paths:
            if os.path.exists(path):
                t = os.path.getmtime(path)
            else:
                t = -np.inf
            times.append(t)

        out = max(times)

        return out

    @property
    def max_timestamp(self):
        if self.rel_path_cur != self.rel_path_prev:
            max_timestamp = np.inf
        else:
            max_timestamp = self.timestamp
        for s in self.prereqs + self.static_prereqs:
            max_timestamp = max(max_timestamp, s.max_timestamp)

        return max_timestamp

    @classmethod
    def default_location(cls):
        return os.path.normpath(DEFAULTS.get(cls.config_key(), cls.DEFAULT_LOCATION))

    @classmethod
    def url(cls):
        return cls.URL

    @classmethod
    def parent_resource(cls):
        return cls.PARENT_RESOURCE

    @classmethod
    def config_key(cls):
        return cls.__name__

    @classmethod
    def infer_paths(cls):
        path = ''
        rel_path_prev = None
        rel_path_cur = None

        if os.path.exists(HISTORY_PATH):
            hist = configparser.ConfigParser()
            hist.optionxform = str
            hist.read(HISTORY_PATH)

            paths = hist['settings']
            rel_path_prev = paths.get(cls.config_key(), None)
            if rel_path_prev is not None:
                rel_path_prev = os.path.normpath(rel_path_prev)

        if os.path.exists(CONFIG_PATH):
            cur = configparser.ConfigParser()
            cur.optionxform = str
            cur.read(CONFIG_PATH)

            paths = cur['settings']
            rel_path_cur = paths.get(cls.config_key(), None)
            if rel_path_cur is not None:
                rel_path_cur = os.path.normpath(rel_path_cur)

        if rel_path_cur is None:
            rel_path_cur = os.path.normpath(cls.default_location())

        if cls.parent_resource() is not None:
            path = os.path.join(path, cls.parent_resource().infer_paths()[0])

        path = os.path.join(path, rel_path_cur)

        return path, rel_path_cur, rel_path_prev

    @classmethod
    def ancestor_exists(cls):
        c = cls
        p = cls.parent_resource()
        while p is not None:
            c = p
            p = p.parent_resource()
        return os.path.exists(c.infer_paths()[0])

    def build(self, dry_run=False):
        if not os.path.exists(self.path) and self.ancestor_exists():
            super(ExternalResource, self).build(dry_run=dry_run)

        if not dry_run:
            if self.rel_path_prev != self.rel_path_cur:
                if not os.path.exists(HISTORY_PATH):
                    with open(HISTORY_PATH, 'w') as f:
                        f.write('[settings]\n\n')
                c = configparser.ConfigParser()
                c.optionxform = str
                c.read(HISTORY_PATH)
                paths = c['settings']
                paths[self.config_key()] = self.rel_path_cur
                with open(HISTORY_PATH, 'w') as f:
                    c.write(f)

        return self.body()

# Infer classes from MB static resources
static_classes = create_classes_from_dir(STATIC_RESOURCES_DIR)
for c in static_classes:
    globals()[c.__name__] = c
del static_classes