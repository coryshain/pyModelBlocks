import sys
import subprocess
import pickle

from .util import tostderr


class Content(object):
    def __init__(
            self,
            data=None,
            path=None,
            is_text=True
    ):
        if data is None and path is not None:
            if is_text:
                read_mode = 'r'
            else:
                read_mode = 'rb'
            with open(path, read_mode) as f:
                if is_text:
                    self.data = f.readlines()
                else:
                    self.data = pickle.load(f)
        else:
            self.data = data

        self.is_text = is_text

    def dump(self, buffer=None):
        close_after = False
        if self.is_text:
            write_mode = 'w'
        else:
            write_mode = 'wb'

        if buffer is not None:
            if isinstance(buffer, str):
                buffer = open(buffer, write_mode)
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
            stdin=None,
            stdout=None,
            stderr=sys.stderr,
            out_mode='w'
    ):
        self.cmd = cmd
        self.stdin_src = stdin
        if stdout == 'PIPE':
            stdout = subprocess.PIPE
        self.stdout_src = stdout
        self.stderr_src = stderr
        self.out_mode = out_mode

    @property
    def stdin(self):
        if isinstance(self.stdin_src, str):
            out = open(self.stdin_src, 'r')
        else:
            out = self.stdin_src

        return out

    @property
    def stdout(self):
        if isinstance(self.stdout_src, str):
            out = open(self.stdout_src, self.out_mode)
        else:
            out = self.stdout_src

        return out

    @property
    def stderr(self):
        if isinstance(self.stderr_src, str):
            out = open(self.stderr_src, self.out_mode)
        else:
            out = self.stderr_src

        return out

    def set_stdin(self, stdin):
        self.stdin_src = stdin

    def set_stdout(self, stdout):
        self.stdout_src = stdout

    def set_stderr(self, stderr):
        self.stderr_src = stderr

    def close_connections(
            self,
            stdin=None,
            stdout=None,
            stderr=None
    ):
        if stdin is None and isinstance(self.stdin_src, str):
            stdin.close()
        if stdout is None and isinstance(self.stdout_src, str):
            stdout.close()
        if stderr is None and isinstance(self.stderr_src, str):
            stderr.close()

    def run(self, dry_run=False):
        raise NotImplementedError


class ShellCommand(Command):
    def __init__(
            self,
            cmd,
            args=None,
            stdin=None,
            stdout=None,
            stderr=sys.stderr,
            out_mode='w'
    ):
        super(ShellCommand, self).__init__(cmd, stdin=stdin, stdout=stdout, stderr=stderr, out_mode=out_mode)
        if args is None:
            args = []
        if not isinstance(args, list):
            args = [args]
        self.args = args

    def __str__(self):
        out = ' '.join([self.cmd] + self.args)
        if isinstance(self.stdout_src, str):
            out += ' > %s' % self.stdout_src

        return out

    def run(self, dry_run=False):
        tostderr(str(self) + '\n')

        if not dry_run:
            stdin = self.stdin
            stdout = self.stdout
            stderr = self.stderr
            out = subprocess.Popen(
                [self.cmd] + self.args,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                shell=False
            )
            out.wait()

            self.close_connections(stdin=stdin, stdout=stdout, stderr=stderr)

        return None


class Pipe(Command):
    def __init__(
            self,
            *cmd,
            stdin=None,
            stdout=None,
            stderr=sys.stderr,
            out_mode='w'
    ):
        super(Pipe, self).__init__(cmd, stdin=stdin, stdout=stdout, stderr=stderr, out_mode=out_mode)

    def __str__(self):
        out = '  |\n  '.join([str(c) for c in self.cmd])
        if isinstance(self.stdout_src, str):
            out += ' > %s' % self.stdout_src

        return out

    def run(self, dry_run=False):
        tostderr(str(self) + '\n')

        if not dry_run:
            stdin = self.stdin
            stdout = self.stdout
            stderr = self.stderr

            stdin_cur = stdin
            for i, c in enumerate(self.cmd):
                self.cmd[i].set_stdin(stdin_cur)

                if i == len(self.cmd) - 1:
                    self.cmd[i].set_stdout(stdout)
                else:
                    self.cmd[i].set_stdout(subprocess.PIPE)

                out_cur = c.run(dry_run=dry_run)
                if stdin is not None:
                    stdin.close()
                stdin_cur = out_cur.stdout

            self.close_connections(stdin=stdin, stdout=stdout, stderr=stderr)

        return None


class Dump(Command):
    def __init__(
            self,
            cmd,
            args=None,
            kwargs=None,
            stdin=None,
            stdout=None,
            stderr=sys.stderr,
            out_mode='w',
            descr='data'
    ):
        super(Dump, self).__init__(cmd, stdin=stdin, stdout=stdout, stderr=stderr, out_mode=out_mode)
        if args is None:
            args = []
        if not isinstance(args, list):
            args = [args]
        self.data = self.cmd
        if not isinstance(self.data, Content):
            is_text = out_mode[-1] != 'b'
            self.data = Content(data=self.data, is_text=is_text)
        self.args = args
        self.kwargs = kwargs
        self.descr = descr

    def __str__(self):
        dest = self.stdout_src
        if not isinstance(dest, str):
            dest = 'internal variable'
        out = 'Dump %s to %s' % (self.descr, dest)

        return out

    def run(self, dry_run=False):
        tostderr(str(self) + '\n')

        if not dry_run:
            stdout = self.stdout
            self.data.dump(stdout)

            self.close_connections(stdout=stdout)

        return self.data



