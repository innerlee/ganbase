## Models

This module provides some common GAN models.

### MLP

Multi-layer perceptron.
<!--
```python
import ganbase as gb

img = gb.read_img('test.jpg')
img_ = gb.read_img(img) # nothing will happen, img_ = img
gb.write_img(img, 'out.jpg')
``` -->

### DCGAN

DCGAN Models
<!--
```python
import ganbase as gb

# resize to a given size
gb.resize(img, (1000, 600), return_scale=True)
# resize to the same size of another image
gb.resize_like(img, dst_img, return_scale=False)
# resize by a ratio
gb.resize_by_ratio(img, 0.5)
# resize so that the max edge no longer than 1000, short edge no longer than 800
# without changing the aspect ratio
gb.resize_keep_ar(img, 1000, 800)
# resize to the maximum size
gb.limit_size(img, 400)
``` -->
