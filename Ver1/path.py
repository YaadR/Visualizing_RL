import os

# Create Path object
#   my_path = Path("/Users/UserName/Desktop")

# Append to Path
#   open(my_path / "test.txt")
#   my_path / "test.txt" == os.path.join(py_path, "test.txt")


class Path:
    base: str

    def __init__(self, base_path: str):
        self.base = base_path.__str__()

    def __str__(self):
        return self.base

    def __truediv__(self, b):
        return os.path.join(self, b)

    def __fspath__(self):
        return self.base


# Path to Visualizing_Rl directory
BASE_DIR = Path(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
