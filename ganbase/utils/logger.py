import sys
import os


class Logger(object):
    """
    Log all the outputs into an autocreated file named `logfile.log`.

    usage:

    ```python
    sys.stdout = gb.Logger("my/awsome/dir")
    ```
    """

    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, "logfile.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class ErrorLogger(object):
    """
    Log all the outputs into an autocreated file named `logfile.log`.

    usage:

    ```python
    sys.stderr = gb.ErrorLogger("my/awsome/dir")
    ```
    """

    def __init__(self, path):
        self.terminal = sys.stderr
        self.log = open(os.path.join(path, "logfile.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
