# Introduction

`ganbase` is a miscellaneous set of tools which is helpful for GAN research.
It comprises the following parts.

- Models
- Datasets
- Evaluation
- Visualize
- Utils

Try and start with

```shell
pip install ganbase
```

See [documentation](http://ganbase.readthedocs.io/en/latest) for more features and usage.

## Examples

Stay tuned.

<!-- There are some popular features such as progress visualization, timer, video to frames/frames to videos.


- Progress visualization

    If you want to apply a method to a list of items and track the progress, `track_progress`
    is a good choice. It will display a progress bar to tell the progress and ETA.

    ```python
    import ganbase as gb

    def func(item):
        # do something
        pass

    tasks = [item_1, item_2, ..., item_n]

    gb.track_progress(func, tasks)
    ```

    The output is like the following.
    ![progress](docs/_static/progress.gif)

    There is another method `track_parallel_progress`, which wraps multiprocessing and
    progress visualization.

    ```python
    import ganbase as gb

    def func(item):
        # do something
        pass

    tasks = [item_1, item_2, ..., item_n]

    gb.track_parallel_progress(func, tasks, 8)
    # 8 workers
    ```

- Timer

    It is convinient to computer the runtime of a code block with `Timer`.

    ```python
    import time

    with gb.Timer():
        # simulate some code block
        time.sleep(1)
    ```

    Or try a more flexible way.

    ```python
    timer = gb.Timer()
    # code block 1 here
    print(timer.since_start())
    # code block 2 here
    print(timer.since_last_check())
    print(timer.since_start())
    ```

- Video/Frames conversion

    To split a video into frames.

    ```python
    video = gb.VideoReader('video_file.mp4')
    video.cvt2frames('frame_dir')
    ```
    Besides `cvt2frames`, `VideoReader` wraps many other useful methods to operate a video like a list object, like

    ```
    video = gb.VideoReader('video_file.mp4')
    len(video)  # get total frame number
    video[5]  # get the 6th frame
    for img in video:  # iterate over all frames
        print(img.shape)
    ```

    To generate a video from frames, use the `frames2video` method.

    ```python
    video = gb.frames2video('frame_dir', 'out_video_file.avi', fps=30)
    ```

- Video editing (needs ffmpeg)

    To cut a video.

    ```python
    gb.cut_video('input.mp4', 'output.mp4', start=3, end=10)
    ```

    To join two video clips.

    ```python
    gb.concat_video(['clip1.mp4', 'clip2.mp4'], 'output.mp4')
    ```

    To resize a video.

    ```python
    gb.resize_video('input.mp4', 'resized.mp4', (360, 240))
    # or
    gb.resize_video('input.mp4', 'resized.mp4', ratio=2)
    ```

    To convert the format of a video.

    ```python
    gb.convert_video('input.avi', 'output.mp4', vcodec='h264')
    ``` -->
