# Taken from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s23.html
def add_sys_path(new_path):
    import sys, os

    # Avoid adding nonexistent paths
    if not os.path.exists(new_path):
        return -1

    # Standardize the path. Windows is case-insensitive, so lowercase
    # for definiteness.
    new_path = os.path.abspath(new_path)
    if sys.platform == "win32":
        new_path = new_path.lower()

    # Check against all currently available paths
    for x in sys.path:
        x = os.path.abspath(x)
        if sys.platform == "win32":
            x = x.lower()
        if new_path in (x, x + os.sep):
            return 0
    sys.path.append(new_path)
    return 1


def add_src_to_sys_path():
    from pathlib import Path

    add_sys_path(Path(__file__).parent.parent / "src")
