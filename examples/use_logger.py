#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import ganbase as gb # pylint: disable=C0413

def use_logger(workdir):

    os.system(f'mkdir -p {workdir}/png')
    sys.stdout = gb.Logger(workdir)
    sys.stderr = gb.ErrorLogger(workdir)


