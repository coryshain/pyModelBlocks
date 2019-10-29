import os
import pickle
import time
import numpy as np

from .util import tostderr


class Data(object):
    def __init__(
            self,
            data=None,
            path=None,
            out_mode='w'
    ):
        self.is_text = out_mode[-1] != 'b'
        self.path = path
        self.out_mode = out_mode
        self.timestamp = -np.inf
        self.set_data(data)

    def set_data(self, data):
        if data is None and self.path is not None:
            if self.is_text:
                read_mode = 'r'
            else:
                read_mode = 'rb'
            if os.path.exists(self.path):
                if os.path.isfile(self.path):
                    with open(self.path, read_mode) as f:
                        if self.is_text:
                            data = f.readlines()
                        else:
                            data = pickle.load(f)
                else:
                    data = 'Directory target'
                self.timestamp = os.path.getmtime(self.path)
        elif data is not None:
            self.timestamp = time.time()
        self.data = data

    def dump(self, buffer=None):
        close_after = False

        if buffer is not None:
            if isinstance(buffer, str):
                buffer = open(buffer, self.out_mode)
                close_after = True

            if self.is_text:
                for l in self.data:
                    buffer.write(l)
            else:
                pickle.dump(self.data, buffer)

            if close_after:
                buffer.close()


class Command(object):
    def __init__(
            self,
            cmd,
            path=None,
            out_mode='w'
    ):
        self.cmd = cmd
        self.path = path
        self.out_mode = out_mode

    def run(self, dry_run=False):
        raise NotImplementedError


class ShellCommand(Command):
    def __init__(
            self,
            cmd,
            path=None,
            out_mode='w'
    ):
        super(ShellCommand, self).__init__(cmd, path=path, out_mode=out_mode)

    def __str__(self):
        return self.cmd

    def run(self, dry_run=False):
        tostderr(str(self) + '\n')

        if not dry_run:
            returncode = os.system(str(self))

            assert returncode == 0, 'Shell execution failed with return code %s' % returncode

        return None


class PyCommand(Command):
    def __init__(
            self,
            cmd,
            path=None,
            out_mode='w',
            descr='data'
    ):
        super(PyCommand, self).__init__(cmd, path=path, out_mode=out_mode)
        self.cmd = self.cmd
        self.descr = descr

    def __str__(self):
        dest = self.path
        if not isinstance(dest, str):
            dest = 'internal variable'
        out = 'Dump %s to %s' % (self.descr, dest)

        return out

    def run(self, dry_run=False):
        tostderr(str(self) + '\n')

        data = None
        if not dry_run:
            data = self.cmd()
            data = Data(data=data, out_mode=self.out_mode)

        return data



