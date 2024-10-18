import io
import sys


class SuppressStdout:
    def __init__(self):
        self.original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = io.StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        