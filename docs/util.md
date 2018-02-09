## Utils

This module offers some misc utils.

### Timer

Tick tick tick...

<!--
```python
import ganbase as gb

# load data from a file
data = gb.load('test.json')
data = gb.load('test.yaml')
data = gb.load('test.pickle')
# load data from a file-like object
with open('test.json', 'r') as f:
    data = gb.load(f)

# dump data to a string
json_str = gb.dump(data, format='json')
# dump data to a file with a filename (infer format from file extension)
gb.dump(data, 'out.pickle')
# dump data to a file with a file-like object
with open('test.yaml', 'w') as f:
    data = gb.dump(data, f, format='yaml')
``` -->
