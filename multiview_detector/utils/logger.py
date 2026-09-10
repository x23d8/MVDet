import os
import sys


class Logger(object):
    def __init__(self, fpath=None, append=False):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            self.file = open(fpath, 'a' if append else 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        # The console is owned by the process/test harness, not by this tee.
        # Closing it makes resumed runs and test capture fail unpredictably.
        if self.file is not None and not self.file.closed:
            self.file.close()
