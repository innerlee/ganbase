#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import ganbase as gb # pylint: disable=C0413

os.system(f'mkdir -p temp')
sys.stdout = gb.Logger('temp')
sys.stderr = gb.ErrorLogger('temp')

print('bf')

raise ValueError('err')
