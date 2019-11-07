import sys
import os
import shutil
import itertools
from functools import cmp_to_key
import re
import pickle
import numpy as np
import pandas as pd

if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser

from mb.util.general import tostderr


#####################################
#
# GLOBAL CONSTANTS
#
#####################################

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_RESOURCES_DIR = os.path.join(ROOT_DIR, 'static_resources')
DEFAULT_PATH = os.path.join(ROOT_DIR, 'config', '.defaults.ini')
CONFIG_PATH = os.path.join(ROOT_DIR, 'config', '.config.ini')
HISTORY_PATH = os.path.join(ROOT_DIR, 'config', '.hist.ini')

DEFAULT = configparser.ConfigParser()
DEFAULT.optionxform = str
DEFAULT.read(DEFAULT_PATH)
DEFAULT_SETTINGS = DEFAULT['settings']

if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as f:
        f.write('[settings]\n\n')
USER = configparser.ConfigParser()
USER.optionxform = str
USER.read(CONFIG_PATH)
USER_SETTINGS = USER['settings']

if not os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'w') as f:
        f.write('[settings]\n\n')

HISTORY = configparser.ConfigParser()
HISTORY.optionxform = str
HISTORY.read(HISTORY_PATH)
HISTORY_SETTINGS = HISTORY['settings']

CFLAGS = USER_SETTINGS.get('cflags', DEFAULT_SETTINGS['cflags'])

DELIM = [
    '.',
    '-',
    '_',
    '+'
]
NON_VAR_CHARS = re.compile('\W')
PUNC = [
    ",",
    ".",
    "``",
    "`",
    "--",
    "''",
    "'",
    "...",
    "?",
    "!",
    ":",
    ";",
    "(",
    ")",
    "-RRB-",
    "-LRB-",
    "-LCB-",
    "-RCB-"
]
DEFAULT_SEP = ' '
DEFAULT_NA = 'NaN'





#####################################
#
# UTILITY METHODS
#
#####################################


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
                descr_long = descr + ' directory'
                parent_name_cur = parent_name + class_name
                out += create_classes_from_dir(path, parent_name=parent_name_cur)
            else:
                descr_long = descr + ' static file'

            class_name = parent_name + class_name

            attr_dict = {
                'SUFFIX': os.path.basename(path),
                'DEFAULT_LOCATION': path,
                'DESCR_SHORT': descr,
                'DESCR_LONG': descr_long
            }

            out.append(type(class_name, (StaticResource,), attr_dict))

    return out


def read_data(path, read_mode='r', sep=DEFAULT_SEP):
    data = None
    if read_mode is not None:
        if read_mode == 'pandas':
            if os.path.exists(path):
                data = pd.read_csv(path, sep=sep)
        else:
            is_text = read_mode[-1] != 'b'
            data = None
            if os.path.exists(path):
                if os.path.isfile(path):
                    with open(path, read_mode) as f:
                        if is_text:
                            data = f.readlines()
                        else:
                            data = pickle.load(f)
                else:
                    data = 'Directory target'

    return data


def dump_data(data, buffer=None, write_mode='w', sep=' ', na_rep=DEFAULT_NA):
    if write_mode is not None:
        if write_mode == 'pandas':
            data.to_csv(buffer, sep=sep, index=False, na_rep=na_rep)
        else:
            is_text = write_mode[-1] != 'b'
            close_after = False

            if buffer is not None:
                if isinstance(buffer, str):
                    buffer = open(buffer, write_mode)
                    close_after = True

                if is_text:
                    for l in data:
                        buffer.write(l)
                else:
                    pickle.dump(data, buffer)

                if close_after:
                    buffer.close()


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
    
    :prm ``s``: str; the input string
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
    
    :prm s: ``str``; the input string
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


def prereq_comparator(x, y):
    if isinstance(x, str) and isinstance(y, str):
        out = 0
    elif isinstance(x, str):
        out = 1
    elif isinstance(y, str):
        out = -1
    else:
        if x.has_suffix and not y.has_suffix:
            out = 1
        elif y.has_suffix and not x.has_suffix:
            out = -1
        else:
            if x.has_prefix and not y.has_prefix:
                out = 1
            elif y.has_prefix and not x.has_prefix:
                out = -1
            else:
                if len(x.stem) < len(y.stem):
                    out = -1
                elif len(y.stem) < len(x.stem):
                    out = 1
                else:
                    out = 0

    return out





#####################################
#
# ABSTRACT MB TYPES
#
#####################################


class MBType(object):
    SUFFIX = ''
    MANIP = ''
    PATTERN_PREREQ_TYPES = []
    STATIC_PREREQ_TYPES = []
    ARG_TYPES = []
    CONFIG_KEYS = []
    FILE_TYPE = 'text'  # one of ['text', 'table', 'obj', None], for text, pandas data table, python-readable binary (pickle), or other (non-python-readable) file, respectively
    SEP = DEFAULT_SEP # Used managing separator conventions in tabular data
    PRECIOUS = False

    REPEATABLE_PREREQ = False

    HAS_PREFIX = False
    HAS_SUFFIX = False

    DESCR_SHORT = 'data'
    DESCR_LONG = (
        "Abstract base class for ModelBlocks types.\n"
    )

    def __init__(self, path):
        path = os.path.normpath(path)
        if path.endswith(self.suffix()):
            if self.suffix() != '':
                path = path[:-len(self.suffix())]
        self.directory = os.path.dirname(path)
        self.basename = os.path.basename(path)
        self.stem = self.strip_suffix(self.basename)
        self.path = os.path.join(self.directory, self.basename + self.suffix())

        self.pattern_prereqs_all_paths_src = []
        self.pattern_prereqs_src = []
        self.static_prereqs_src = []
        self.other_prereqs_all_paths_src = []
        self.other_prereqs_src = []
        self.dependencies = []

        self.has_prefix_src = False
        self.has_suffix_src = False

        self.data = None

        self.dump = self.precious() or os.path.exists(self.path)

        self.graph = None
        self.intermediate = not self.dump

        self.fn_dry_run = None
        self.fn = None
        self.built = False
        self.process_scheduler = None

    @property
    def args(self):
        args = self.parse_path(self.path, has_prefix=self.has_prefix, has_suffix=self.has_suffix)

        if args is not None:
            if 'basename' in args:
                del args['basename']
            if 'prereqs' in args:
                del args['prereqs']
            if 'static_prereqs' in args:
                del args['static_prereqs']

        return args

    @property
    def has_prefix(self):
        return self.has_prefix_src

    @property
    def has_suffix(self):
        return self.has_suffix_src

    @property
    def timestamp(self):
        return get_timestamp(self.path)

    @property
    def max_timestamp(self):
        max_timestamp = self.timestamp
        for k, old, new in self.config_values():
            if old != new:
                max_timestamp = np.inf
                break
        if max_timestamp < np.inf:
            for s in self.pattern_prereqs_src + self.static_prereqs_src + self.other_prereqs_src:
                max_timestamp = max(max_timestamp, s.max_timestamp)

        return max_timestamp

    @property
    def output_buffer(self):
        if self.dump:
            out = self.path
        else:
            out = None

        return out

    @property
    def concurrent(self):
        return self.process_scheduler is not None

    @classmethod
    def suffix(cls):
        return cls.SUFFIX

    @classmethod
    def manip(cls):
        return cls.MANIP

    @classmethod
    def pattern_prereq_types(cls):
        return cls.PATTERN_PREREQ_TYPES
    
    @classmethod
    def augment_prereq(cls, i, path):
        return ''

    @classmethod
    def has_multiple_pattern_prereqs(cls):
        return len(cls.pattern_prereq_types()) > 1 or (len(cls.pattern_prereq_types()) == 1 and cls.repeatable_prereq())

    @classmethod
    def static_prereq_types(cls):
        return cls.STATIC_PREREQ_TYPES

    @classmethod
    def other_prereq_paths(cls, path):
        return []

    @classmethod
    def repeatable_prereq(cls):
        return cls.REPEATABLE_PREREQ

    @classmethod
    def arg_types(cls):
        return cls.ARG_TYPES

    @classmethod
    def config_keys(cls):
        out = []
        for x in cls.CONFIG_KEYS:
            try:
                key, default = x
            except TypeError:
                key = x
                default = None
            out.append((key, default))
        return out

    @classmethod
    def config_values(cls):
        out = []
        for key, default in cls.config_keys():
            val_prev = HISTORY_SETTINGS.get(key, None)
            if val_prev is not None and key.endswith('_path'):
                val_prev = os.path.normpath(val_prev)

            val_cur = USER_SETTINGS.get(
                key,
                DEFAULT_SETTINGS.get(
                    key,
                    default
                )
            )
            if val_cur is not None and key.endswith('_path'):
                val_cur = os.path.normpath(val_cur)
            out.append((key, val_prev, val_cur))

        return out

    @classmethod
    def read_mode(cls):
        file_type = cls.file_type()
        if file_type == 'text':
            return 'r'
        elif file_type == 'table':
            return 'pandas'
        elif file_type == 'python':
            return 'rb'
        elif file_type == None:
            return None
        else:
            raise ValueError("Unrecognized file type %s. Must be one of ['text', 'python', None]." % file_type)

    @classmethod
    def write_mode(cls):
        file_type = cls.file_type()
        if file_type == 'text':
            return 'w'
        elif file_type == 'table':
            return 'pandas'
        elif file_type == 'python':
            return 'wb'
        elif file_type == None:
            return None
        else:
            raise ValueError("Unrecognized file type %s. Must be one of ['text', 'python', None]." % file_type)

    @classmethod
    def file_type(cls):
        return cls.FILE_TYPE

    @classmethod
    def sep(cls):
        return cls.SEP

    @classmethod
    def precious(cls):
        return cls.PRECIOUS

    @classmethod
    def is_text(cls):
        return cls.FILE_TYPE == 'text'

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
            out |= c.inheritors()

        return out

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'MBType'

    @classmethod
    def match(cls, path):
        out = not cls.is_abstract()
        if out:
            suffix = cls.manip() + cls.suffix()
            out = path.endswith(suffix)
            if out:
                prereq_types = cls.pattern_prereq_types()
                if len(prereq_types) == 0:
                    out = len(os.path.basename(path)) == len(suffix)
                elif len(prereq_types) > 1 or cls.repeatable_prereq():
                    basenames = path[:-len(suffix)].split(DELIM[0])
                    prereq_types = prereq_types[:]
                    if cls.repeatable_prereq():
                        while len(prereq_types) < len(basenames):
                            prereq_types.insert(0, prereq_types[0])

                    out = 0 <= len(basenames) - len(prereq_types) <= 2

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
    def parse_args(cls, path):
        basename = cls.strip_suffix(path)
        out = {'basename': basename}
        if len(cls.arg_types()) > 0:
            directory = os.path.dirname(basename)
            basename_split = os.path.basename(basename).split(DELIM[0])
            basename = DELIM[0].join(basename_split[:-1])
            out = {'basename': os.path.join(directory, basename)}
            argstr = basename_split[-1]
            argstr = argstr.split(DELIM[1])
            args = [a for a in cls.arg_types() if a.positional]
            kwargs = [a for a in cls.arg_types() if not a.positional]
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

                assert out_cur is not None, 'Unrecognized keyword argument %s' % s
                out.update(out_cur)

            for kwarg in kwargs:
                out[kwarg.key] = kwarg.default

        return out

    @classmethod
    def parse_path(cls, path, has_prefix=False, has_suffix=False):
        out = None
        path = os.path.normpath(path)
        if cls.match(path):
            out = cls.parse_args(path)
            out['prereqs'] = []
            prereqs = []
            if cls.has_multiple_pattern_prereqs():
                basename = out['basename']
                directory = os.path.dirname(basename)
                basename = os.path.basename(basename)
                basenames = basename.split(DELIM[0])
                if has_prefix:
                    prefix = decrement_delimiters(basenames[0])
                    basenames = basenames[1:]
                else:
                    prefix = ''
                if len(basenames) == 0:
                    out = None
                else:
                    if has_suffix:
                        suffix = decrement_delimiters(basenames[-1])
                        basenames = basenames[:-1]
                    else:
                        suffix = ''
                    if len(basenames) == 0:
                        out = None
                    else:
                        basenames = [os.path.join(directory, DELIM[0].join(
                            [y for y in (prefix, decrement_delimiters(x), suffix) if y != ''])) for x in basenames]
                        prereq_types = cls.pattern_prereq_types()[:]
                        if cls.repeatable_prereq():
                            while len(prereq_types) < len(basenames):
                                prereq_types.insert(0, prereq_types[0])
                        for b, p in zip(basenames, prereq_types):
                            prereq_path = b
                            prereq_path += cls.augment_prereq(0, path)
                            prereq_path += p.suffix()
                            prereqs.append(prereq_path)
                        out['prereqs'] = prereqs
            elif len(cls.pattern_prereq_types()) == 1:
                prereq_path = out['basename']
                prereq_path += cls.augment_prereq(0, path)
                prereq_path += cls.pattern_prereq_types()[0].suffix()
                prereqs.append(prereq_path)
                out['prereqs'] = prereqs

            if out is not None:
                out['has_prefix'] = has_prefix
                out['has_suffix'] = has_suffix

        return out

    @classmethod
    def report_api(cls):
        out = '-' * 50 + '\n'
        out += 'Class:                %s\n' % cls.__name__
        if issubclass(cls, ExternalResource) and cls.url() is not None:
            out += 'URL:                  %s\n' % cls.url()
        out += 'Short description:    %s\n' % cls.descr_short()
        out += 'Detailed description: %s\n' % cls.descr_long()
        external_resources = [x for x in cls.static_prereq_types() if issubclass(x, ExternalResource)]
        if len(external_resources) > 0:
            out += 'External resources:\n'
            for x in external_resources:
                out += '  %s\n' % x.__name__
        out += 'Prerequisites:\n'
        for i, x in enumerate(cls.pattern_prereq_types()):
            out += '  %s' % x.__name__
            if i == 0 and cls.repeatable_prereq():
                out += ' (repeatable)'
            out += '\n'
        for x in cls.other_prereq_paths(None):
            out += '  %s\n' % x
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
            for i, x in enumerate(cls.pattern_prereq_types()):
                name = x.__name__
                if i == 0 and cls.repeatable_prereq():
                    s = '<%s>(.<%s>)+' % (name, name)
                else:
                    s = '<%s>' % name
                out.append(s)
            if cls.has_suffix():
                out.append('<SUFFIX>')
            out = '(<DIR>/)' + DELIM[0].join(out)
            if len(cls.arg_types()) > 0:
                out += DELIM[0]
                arg_str = ['<%s>' % a.syntax_str() for a in cls.arg_types()]
                out += DELIM[1].join(arg_str)
            out += cls.manip()
            out += cls.suffix()

        return out

    def pattern_prereqs_all_paths(self):
        return self.pattern_prereqs_all_paths_src

    def pattern_prereqs(self):
        return self.pattern_prereqs_src

    def other_prereqs_all_paths(self):
        return self.other_prereqs_all_paths_src

    def other_prereqs(self):
        return self.other_prereqs_src

    def static_prereqs(self):
        return self.static_prereqs_src

    def set_data(self, data=None):
        read_mode = self.read_mode()
        if read_mode is not None:
            if data is None:
                data = read_data(self.path, read_mode=read_mode, sep=self.sep())
        self.data = data

    def body(self):
        raise NotImplementedError

    def body_args(self, dry_run=False):
        out = self.pattern_prereq_data(dry_run=dry_run) + self.other_prereq_data(dry_run=dry_run)
        args = self.args
        for a in self.arg_types():
            out.append(args[a.key])
        out = tuple(out)

        return out

    def pattern_prereq_data(self, dry_run=False):
        return [x.get(dry_run=dry_run) for x in self.pattern_prereqs()]

    def static_prereq_data(self, dry_run=False):
        return [x.get(dry_run=dry_run) for x in self.static_prereqs()]

    def other_prereq_data(self, dry_run=False):
        return [x.get(dry_run=dry_run) for x in self.other_prereqs()]

    def get(self, dry_run=False, force=False, report_up_to_date=False):
        build = force or (self.max_timestamp > self.timestamp)
        if build and not self.built:
            body = self.body()
            args = self.body_args(dry_run=dry_run)
            if not dry_run:
                self.set_data()

            if isinstance(self.body(), str):
                descr = body
                mode = self.read_mode()

                if dry_run:
                    def fn(body, descr, dump, path, mode, *args):
                        tostderr(descr + '\n')

                        return None

                else:
                    def fn(body, descr, dump, path, mode, *args):
                        tostderr(descr + '\n')

                        returncode = os.system(body)
                        assert returncode == 0, 'Shell execution failed with return code %s' % returncode

                        data = read_data(path, read_mode=mode, sep=self.sep())

                        return data
            elif hasattr(body, '__call__'):
                if self.dump:
                    descr = 'Computing and dumping %s' % self.path
                else:
                    descr = 'Computing and storing %s' % self.path

                mode = self.write_mode()

                if dry_run:
                    def fn(body, descr, dump, path, mode, *args):
                        tostderr(descr + '\n')

                        return None
                else:
                    def fn(body, descr, dump, path, mode, *args):
                        tostderr(descr + '\n')

                        data = body(*args)

                        if dump:
                            dump_data(data, buffer=path, write_mode=mode)

                        return data

            else:
                descr = None
                mode = None
                body = self.data
                
                def fn(body, descr, dump, path, mode, *args):
                    return body

            if self.concurrent:
                fn = self.process_scheduler.remote(fn).remote

            data = fn(
                body,
                descr,
                self.dump,
                self.path,
                mode,
                *args
            )

            self.set_data(data)

            self.built = True
        else:
            if dry_run:
                def fn(path, data, report_up_to_date=False):
                    if report_up_to_date:
                        tostderr('%s is up to date.\n' % path)

                    return None
            else:
                def fn(path, data, report_up_to_date=False):
                    if report_up_to_date:
                        tostderr('%s is up to date.\n' % path)

                    return data

            if self.concurrent:
                fn = self.process_scheduler.remote(fn).remote

            if self.data is None and not dry_run:
                self.set_data()

            data = fn(self.path, self.data, report_up_to_date=report_up_to_date)

        return data

    def get_garbage(self, force=False):
        garbage = set()
        build = force or (self.max_timestamp > self.timestamp)

        if build:
            for s in self.pattern_prereqs() + self.other_prereqs():
                garbage |= s.get_garbage(force=force)

            if self.intermediate and self.dump and not self.precious():
                garbage.add(self.path)

        return garbage

    def update_history(self):
        for x in self.static_prereqs_src + self.pattern_prereqs() + self.other_prereqs():
            x.update_history()
        for k, _, v in self.config_values():
            HISTORY_SETTINGS[k] = v

    def set_pattern_prereqs(self, prereqs):
        self.pattern_prereqs_all_paths_src = prereqs

        # assert self.pattern_prereqs_all_paths_src is not None, 'No recipe to make %s' % self.path
        prereqs = []
        if self.pattern_prereqs_all_paths_src is not None:
            for p in self.pattern_prereqs_all_paths_src:
                candidates = sorted(list(p), key=cmp_to_key(prereq_comparator))
                if len(candidates) > 0:
                    prereqs.append(candidates[0])
                else:
                    prereqs.append(None)

        self.pattern_prereqs_src = prereqs

    def set_static_prereqs(self, prereqs):
        self.static_prereqs_src = prereqs
        # assert self.static_prereqs_src is not None, 'No recipe to make %s' % self.path

    def set_other_prereqs(self, prereqs):
        self.other_prereqs_all_paths_src = prereqs

        # assert self.other_prereqs_all_paths_src is not None, 'No recipe to make %s' % self.path
        prereqs = []
        if self.other_prereqs_all_paths_src is not None:
            for p in self.other_prereqs_all_paths_src:
                candidates = sorted(list(p), key=cmp_to_key(prereq_comparator))
                prereqs.append(candidates[0])

        self.other_prereqs_src = prereqs

    def set_dump(self):
        dump = False
        if isinstance(self.body(), str):  # is a shell command and needs to be dumped
            dump = True
            self.dump = dump
        if dump:
            for p in self.pattern_prereqs_all_paths_src:
                for q in p:
                    q.dump = True
            for q in self.static_prereqs_src:
                q.dump = True
            for p in self.other_prereqs_all_paths_src:
                for q in p:
                    q.dump = True

    def exists(self):
        return self.data.data is not None

    def directories_to_make(self, force=False):
        out = set()
        build = force or (self.max_timestamp > self.timestamp)
        if build:
            if self.dump and len(self.directory) > 0 and not os.path.exists(self.directory):
                out.add(self.directory)
            for p in self.pattern_prereqs() + self.static_prereqs() + self.other_prereqs():
                out |= p.directories_to_make(force=force)
        return out


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
    def is_abstract(cls):
        return cls.__name__ == 'StaticResource'

    @classmethod
    def match(cls, path):
        out = os.path.abspath(path) == os.path.abspath(cls.infer_paths()[0])
        return out

    def body(self):
        return self.data



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
        self.stem = self.strip_suffix(self.basename)
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

    @classmethod
    def default_location(cls):
        return os.path.normpath(DEFAULT_SETTINGS.get(cls.__name__ + '_path', cls.DEFAULT_LOCATION))

    @classmethod
    def url(cls):
        return cls.URL

    @classmethod
    def parent_resource(cls):
        return cls.PARENT_RESOURCE

    @classmethod
    def config_keys(cls):
        out = [(cls.__name__ + '_path', cls.default_location())]
        for x in cls.CONFIG_KEYS:
            try:
                key, default = x
            except TypeError:
                key = x
                default = None
            out.append((key, default))
        return out

    @classmethod
    def infer_paths(cls):
        path = ''

        _, rel_path_prev, rel_path_cur = cls.config_values()[0]

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

    @classmethod
    def is_abstract(cls):
        return cls.__name__ == 'ExternalResource'

    def body_args(self, dry_run=False):
        out = self.static_prereq_data(dry_run=dry_run)

        return out





#####################################
#
# CORE MB TYPES
#
#####################################


class MBFailure(MBType):
    DESCR_SHORT = 'failed target'
    DESCR_LONG = (
        "Class to represent build failures.\n"
    )

    def __init__(self, path, cls):
        super(MBFailure, self).__init__(path)
        self.cls = cls

    def cls_suffix(self):
        return self.cls.suffix()

    def cls_manip(self):
        return self.cls.manip()

    def cls_pattern_prereq_types(self):
        return self.cls.pattern_prereq_types()

    def cls_static_prereq_types(self):
        return self.cls.static_prereq_types()

    def cls_other_prereq_paths(self):
        return self.cls.cls_other_prereq_paths(self.path)


class FoundFile(MBType):
    DESCR_SHORT = 'found file'
    DESCR_LONG = (
        "Class to represent found files.\n"
        "A found file cannot be built but exists at the specified path.\n"
        "Found files are always treated as old.\n"
        "They are also given lowest precedence for dependency path selection.\n"
    )

    @property
    def max_timestamp(self):
        return -np.inf


class ParamFile(MBType):
    SUFFIX = 'prm.ini'
    DESCR_SHORT = 'param file'
    DESCR_LONG = (
        "File containing configuration parameters for building targets.\n"
    )

    @classmethod
    def match(cls, path):
        return path.endswith(cls.suffix())

    @classmethod
    def parse_path(cls, path, has_prefix=False, has_suffix=False):
        out = None
        path = os.path.normpath(path)
        if cls.match(path):
            out = cls.parse_args(path)
            out['prereqs'] = []
            out['src'] = os.path.join(
                ROOT_DIR,
                'static_resources',
                'prm',
                os.path.basename(out['basename']) + cls.suffix()
            )

        return out
    
    @classmethod
    def other_prereq_paths(cls, path):
        out = [cls.parse_path(path)['src']]
        
        return out

    def body(self):
        out = 'cp %s %s' % (
            self.other_prereqs()[0].path,
            self.path
        )
        
        return out
        
    
    




#####################################
#
# OTHER TYPES
#
#####################################


class Arg(object):
    """
    Object representing an argument to a data transform (positional or keyword).
    Arguments parameterize all aspects of the transform except the input data,
    which should be computed as prerequisites in the dependency graph.
    Positional arguments are treated as obligatory, keyword arguments are treated as optional.

    :prm key: ``str``; name of the argument
    :prm dtype: ``type``; data type of the argument
    :prm positional: ``bool``; whether the argument is positional
    :prm default: default value for the argument
    """

    def __init__(
            self,
            key,
            dtype=str,
            positional=False,
            default=None,
            descr=None
    ):
        self.key = key
        self.dtype = dtype
        self.positional = positional
        self.default = default
        if descr is None:
            descr = key
        self.descr = descr

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


class SuccessSet(set):
    def add(self, other):
        assert not isinstance(other, MBFailure), 'Cannot add a failed target to SuccessSet object'
        super(SuccessSet, self).add(other)


class FailureSet(set):
    def add(self, other):
        assert isinstance(other, MBFailure), 'FailureSet object can only contain members of type MBFailure'
        super(FailureSet, self).add(other)


class Graph(object):
    def __init__(self, target_paths=None, process_scheduler=None):
        self.target_paths = None
        self.targets_all_paths = None
        self.targets = None
        self.failed_targets_all_paths = None
        self.failed_targets_all_paths = None
        self.process_scheduler = process_scheduler
        self.nodes = {}

        self.build(target_paths)

    def __iter__(self):
        return self.nodes.__iter__()

    def __setitem__(self, key, value):
        self.nodes[key] = value

    def __getitem__(self, key):
        return self.nodes.get(key, None)

    @property
    def concurrent(self):
        return self.process_scheduler is not None

    def build(self, targets):
        tostderr('Building dependency graph...\n')

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

            self.build_graph()

        if len(self.failed_targets_all_paths) > 0:
            report = 'The following targets failed and will be skipped:\n'
            max_num_len = len(str(len(self.failed_target_paths)))
            for i, t in enumerate(self.failed_target_paths):
                num_pad = max_num_len - len(str(i)) + 1
                report += '  ' + '%d. ' % (i + 1) + ' ' * num_pad + '%s\n' % t
            report += '\nAttempted dependency paths:\n'
            report += '(numbers index dependencies, dashes delimit alternatives)\n'
            report += self.report_failure(self.failed_targets_all_paths, indent=2) + '\n'

            tostderr(report)

    def build_graph(self):
        # Compute all dependency paths to target set
        targets_all_paths = []
        for target_path in self.target_paths:
            successes = SuccessSet()
            failures = FailureSet()
            inheritors = MBType.inheritors() - {MBFailure, FoundFile}

            for c in inheritors:
                if not c.is_abstract():
                    if c.has_multiple_pattern_prereqs():
                        parse_settings = list(itertools.product([False, True], [False, True]))
                    else:
                        parse_settings = [(False, False)]

                    for parse_setting in parse_settings:
                        has_prefix_cur, has_suffix_cur = parse_setting
                        subgraph = self.build_subgraph(
                            c,
                            target_path,
                            has_prefix=has_prefix_cur,
                            has_suffix=has_suffix_cur
                        )
                        pat = subgraph['pattern_prereqs_all_paths']
                        stat = subgraph['static_prereqs']
                        oth = subgraph['other_prereqs_all_paths']
                        success_ratio_cur = subgraph['success_ratio']
                        match_cur = subgraph['match']

                        if success_ratio_cur == 1:
                            p = self[(c, target_path, has_prefix_cur, has_suffix_cur)]
                            if p is None:
                                if issubclass(c, StaticResource):
                                    p = c()
                                else:
                                    p = c(target_path)
                                self.add_node(p)
                                p.set_pattern_prereqs(pat)
                                p.set_static_prereqs(stat)
                                p.set_other_prereqs(oth)
                                p.has_prefix_src = has_prefix_cur
                                p.has_suffix_src = has_suffix_cur
                            p.dump = True
                            p.set_dump()
                            p.intermediate = False
                            successes.add(p)
                        elif match_cur:
                            p = MBFailure(target_path, c)
                            p.set_pattern_prereqs(pat)
                            p.set_static_prereqs(stat)
                            p.set_other_prereqs(oth)
                            p.has_prefix_src = has_prefix_cur
                            p.has_suffix_src = has_suffix_cur
                            failures.add(p)

            if len(successes) > 0:
                to_append = successes
            else:
                if os.path.exists(target_path):
                    to_append = SuccessSet({FoundFile(target_path)})
                elif len(failures) > 0:
                    to_append = failures
                else:
                    to_append = FailureSet([target_path])

            targets_all_paths.append(to_append)

        # Filter out failed dependency paths, report problems if none succeed
        exists = [True if isinstance(x, SuccessSet) else False for x in targets_all_paths]

        targets_all_paths_tmp = targets_all_paths
        targets_all_paths = []
        failed_targets_all_paths = []
        failed_target_paths = []
        for i, x in enumerate(exists):
            if x:
                targets_all_paths.append(targets_all_paths_tmp[i])
            else:
                failed_targets_all_paths.append(targets_all_paths_tmp[i])
                failed_target_paths.append(self.target_paths[i])

        self.targets_all_paths = targets_all_paths
        self.targets = []
        for t in self.targets_all_paths:
            candidates = sorted(list(t), key=cmp_to_key(prereq_comparator))
            self.targets.append(candidates[0])
        self.failed_targets_all_paths = failed_targets_all_paths
        self.failed_target_paths = failed_target_paths


    def build_subgraph(self, cls, path, has_prefix=False, has_suffix=False, downstream_paths=None):
        if downstream_paths is None:
            downstream_paths = set()
        downstream_paths.add(path)
        pattern_prereqs_all_paths = None
        static_prereqs = None
        other_prereqs_all_paths = None

        parsed = cls.parse_path(path, has_prefix=has_prefix, has_suffix=has_suffix)
        if parsed is not None:
            # STATIC PREREQS
            static_prereq_types = cls.static_prereq_types()[:]
            static_prereqs = []
            for c in static_prereq_types:
                prereq_path = c.infer_paths()[0]
                subgraph = self.build_subgraph(c, prereq_path)
                pat = subgraph['pattern_prereqs_all_paths']
                stat = subgraph['static_prereqs']
                oth = subgraph['other_prereqs_all_paths']

                p = self[(c, prereq_path, False, False)]
                if p is None:
                    p = c()
                    self.add_node(p)
                    p.set_pattern_prereqs(pat)
                    p.set_static_prereqs(stat)
                    p.set_other_prereqs(oth)
                static_prereqs.append(p)
            static_prereqs = static_prereqs

            has_buildable_prereqs = (len(cls.pattern_prereq_types()) > 0) or (len(cls.other_prereq_paths(path)) > 0)

            if has_buildable_prereqs:
                # PATTERN PREREQS
                pattern_prereq_types = cls.pattern_prereq_types()[:]
                if cls.repeatable_prereq():
                    while len(pattern_prereq_types) < len(parsed['prereqs']):
                        pattern_prereq_types.insert(0, pattern_prereq_types[0])
    
                pattern_prereqs_all_paths = []
                for (P, prereq_path) in zip(pattern_prereq_types, parsed['prereqs']):
                    successes = SuccessSet()
                    failures = FailureSet()
                    inheritors = (P.inheritors() | {P}) - {MBFailure, FoundFile}

                    if prereq_path not in downstream_paths:
                        for c in inheritors:
                            if not c.is_abstract():
                                if c.has_multiple_pattern_prereqs():
                                    parse_settings = list(itertools.product([False, True], [False, True]))
                                else:
                                    parse_settings = [(False, False)]

                                for parse_setting in parse_settings:
                                    has_prefix_cur, has_suffix_cur = parse_setting
                                    subgraph = self.build_subgraph(
                                        c,
                                        prereq_path,
                                        has_prefix=has_prefix_cur,
                                        has_suffix=has_suffix_cur,
                                        downstream_paths=downstream_paths.copy()
                                    )
                                    pat = subgraph['pattern_prereqs_all_paths']
                                    stat = subgraph['static_prereqs']
                                    oth = subgraph['other_prereqs_all_paths']
                                    success_ratio_cur = subgraph['success_ratio']
                                    match_cur = subgraph['match']

                                    if success_ratio_cur == 1:
                                        p = self[(c, prereq_path, has_prefix_cur, has_suffix_cur)]
                                        if p is None:
                                            if issubclass(c, StaticResource):
                                                p = c()
                                            else:
                                                p = c(prereq_path)
                                            self.add_node(p)
                                            p.set_pattern_prereqs(pat)
                                            p.set_static_prereqs(stat)
                                            p.set_other_prereqs(oth)
                                            p.has_prefix_src = has_prefix_cur
                                            p.has_suffix_src = has_suffix_cur
                                        p.set_dump()
                                        successes.add(p)
                                    elif match_cur:
                                        p = MBFailure(prereq_path, c)
                                        p.set_pattern_prereqs(pat)
                                        p.set_static_prereqs(stat)
                                        p.set_other_prereqs(oth)
                                        p.has_prefix_src = has_prefix_cur
                                        p.has_suffix_src = has_suffix_cur
                                        failures.add(p)

                    if len(successes) > 0:
                        to_append = successes
                    else:
                        if os.path.exists(prereq_path):
                            to_append = SuccessSet({FoundFile(prereq_path)})
                        elif len(failures) > 0:
                            to_append = failures
                        else:
                            if prereq_path not in downstream_paths:
                                name = prereq_path
                            else:
                                name = prereq_path + ' (cyclic dependency)'
                            to_append = FailureSet([name])

                    pattern_prereqs_all_paths.append(to_append)
    
                # OTHER PREREQS
                other_prereq_paths = cls.other_prereq_paths(path)
                other_prereqs_all_paths = []
                for prereq_path in other_prereq_paths:
                    prereq_path = os.path.normpath(prereq_path)
                    successes = SuccessSet()
                    failures = FailureSet()
                    inheritors = MBType.inheritors() - {MBFailure, FoundFile}

                    if prereq_path not in downstream_paths:
                        for c in inheritors:
                            if not c.is_abstract():
                                if c.has_multiple_pattern_prereqs():
                                    parse_settings = list(itertools.product([False, True], [False, True]))
                                else:
                                    parse_settings = [(False, False)]

                                for parse_setting in parse_settings:
                                    has_prefix_cur, has_suffix_cur = parse_setting
                                    subgraph = self.build_subgraph(
                                        c,
                                        prereq_path,
                                        has_prefix=has_prefix_cur,
                                        has_suffix=has_suffix_cur,
                                        downstream_paths=downstream_paths.copy()
                                    )
                                    pat = subgraph['pattern_prereqs_all_paths']
                                    stat = subgraph['static_prereqs']
                                    oth = subgraph['other_prereqs_all_paths']
                                    success_ratio_cur = subgraph['success_ratio']
                                    match_cur = subgraph['match']

                                    if success_ratio_cur == 1:
                                        p = self[(c, prereq_path, has_prefix_cur, has_suffix_cur)]
                                        if p is None:
                                            if issubclass(c, StaticResource):
                                                p = c()
                                            else:
                                                p = c(prereq_path)
                                            self.add_node(p)
                                            p.set_pattern_prereqs(pat)
                                            p.set_static_prereqs(stat)
                                            p.set_other_prereqs(oth)
                                            p.has_prefix_src = has_prefix_cur
                                            p.has_suffix_src = has_suffix_cur
                                        p.set_dump()
                                        successes.add(p)
                                    elif match_cur:
                                        p = MBFailure(prereq_path, c)
                                        p.set_pattern_prereqs(pat)
                                        p.set_static_prereqs(stat)
                                        p.set_other_prereqs(oth)
                                        p.has_prefix_src = has_prefix_cur
                                        p.has_suffix_src = has_suffix_cur
                                        failures.add(p)

                    if len(successes) > 0:
                        to_append = successes
                    else:
                        if os.path.exists(prereq_path):
                            to_append = SuccessSet({FoundFile(prereq_path)})
                        elif len(failures) > 0:
                            to_append = failures
                        else:
                            if prereq_path not in downstream_paths:
                                name = prereq_path
                            else:
                                name = prereq_path + ' (cyclic dependency)'
                            to_append = FailureSet([name])

                    other_prereqs_all_paths.append(to_append)
            else:
                pattern_prereqs_all_paths = []
                other_prereqs_all_paths = []

            success_ratio_num = 0
            success_ratio_denom = 0
            for x in pattern_prereqs_all_paths + other_prereqs_all_paths:
                if isinstance(x, SuccessSet):
                    success_ratio_num += 1
                success_ratio_denom += 1
    
            if success_ratio_denom > 0:
                success_ratio = success_ratio_num / success_ratio_denom
            else:
                success_ratio = 1
        else:
            success_ratio = 0

        match = cls.match(path)

        return {
            'pattern_prereqs_all_paths': pattern_prereqs_all_paths,
            'static_prereqs': static_prereqs,
            'other_prereqs_all_paths': other_prereqs_all_paths,
            'success_ratio': success_ratio,
            'match': match
        }

    def add_node(self, node):
        assert not (type(node), node.path) in self.nodes, 'Attempted to re-insert an existing key: %s' % str((type(node), node.path))
        k = (type(node), node.path, node.has_prefix, node.has_suffix)
        v = node
        self[k] = v
        node.graph = self
        node.process_scheduler = self.process_scheduler

    def report_failure(self, targets, indent=0):
        out = ''

        affixes = ['prefix', 'suffix']

        max_num_len = len(str(len(targets)))
        for i, x in enumerate(targets):
            num_pad = max_num_len - len(str(i)) + 1
            if isinstance(x, FailureSet):
                for j, y in enumerate(x):
                    if isinstance(y, str):
                        out += ' ' * (indent) + '%d.' % (i + 1) + ' ' * num_pad + '- FAIL: ' + y + '\n'
                        out += ' ' * (indent + max_num_len + 10) + 'Path does not match any existing constructor\n'
                    else:
                        type_name = [y.cls.__name__]
                        if y.has_prefix:
                            type_name.append('prefix')
                        if y.has_suffix:
                            type_name.append('suffix')
                        type_name = ', '.join(type_name)

                        if j == 0:

                            out += ' ' * (indent) + '%d.' % (i+1) + ' ' * num_pad + '- FAIL: ' + y.path + ' (%s)\n' % type_name
                        else:
                            out += ' ' * (indent) + ' ' * (max_num_len + 2) + '- FAIL: ' + y.path + ' (%s)\n' % type_name

                        out += self.report_failure(y.pattern_prereqs_all_paths() + y.other_prereqs_all_paths(), indent=indent+num_pad+4)
            elif isinstance(x, SuccessSet):
                if len(x) > 0:
                    for j, y in enumerate(x):
                        type_name = [y.__class__.__name__]
                        if y.has_prefix:
                            type_name.append('prefix')
                        if y.has_suffix:
                            type_name.append('suffix')
                        type_name = ', '.join(type_name)

                        if j == 0:
                            out += ' ' * (indent) + '%d.' % (i+1) + ' ' * num_pad + '- PASS: ' + y.path + ' (%s)\n' % type_name
                        else:
                            out += ' ' * (indent) + '-' + ' ' * (max_num_len + 2) + '- PASS: ' + y.path + ' (%s)\n' % type_name
            elif isinstance(x, str):
                out += ' ' * (indent) + '%d.' % (i+1) + ' ' * num_pad + '- FAIL: ' + x + '\n'
                out += ' ' * (indent + max_num_len + 10) + 'Path does not match any existing constructor\n'

        return out

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

            prereq_sets |= deps
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

    def get_garbage(self, force=False):
        garbage = set()
        for t in self.targets:
            garbage |= t.get_garbage(force=force)

        return garbage

    def update_history(self):
        for target in self.targets:
            target.update_history()
        with open(HISTORY_PATH, 'w') as f:
            HISTORY.write(f)

    def get(self, dry_run=False, force=False):
        # Make any needed directories
        directories_to_make = set()
        for t in self.targets:
            directories_to_make |= t.directories_to_make(force=force)
        directories_to_make = sorted(list(directories_to_make))
        if len(directories_to_make) > 0:
            tostderr('Making directories:\n')
            for d in directories_to_make:
                tostderr('  %s\n' % d)
                if not dry_run:
                    os.makedirs(d)

        # Compute list of garbage to collect
        garbage = sorted(list(self.get_garbage(force=force)))

        # Run targets
        tostderr('Running data...\n')
        out = [x.get(dry_run=dry_run, force=force, report_up_to_date=True) for x in self.targets]

        # Update history (links to external resources)
        if not dry_run:
            self.update_history()

        if self.concurrent:
            out = self.process_scheduler.get(out)

        # Clean up intermediate targets
        if len(garbage) > 0:
            tostderr('Garbage collecting intermediate files and directories:\n')
            for p in garbage:
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

        return out





#####################################
#
# INFER STATIC RESOURCE TYPES
#
#####################################


static_classes = create_classes_from_dir(STATIC_RESOURCES_DIR)
for c in static_classes:
    globals()[c.__name__] = c
del static_classes

