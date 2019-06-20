# -*- coding: utf-8 -*-


import os
import errno


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def walk_all_files_with_suffix(dir, suffixs=('.jpg', '.png')):
    paths = []
    names = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            if os.path.splitext(fname)[1] in suffixs:
                paths.append(path)
                names.append(fname)
    return len(names), names, paths
