# Video-decorruption
This reposetory tries to find a logic sequence from randomized frames using VGG16 features

### Dependencies

- Keras
- tqdm
- sklearn
- numpy
- OpenCV

### Data structure

In order to create the correct data structure, use the following commands:

```
$ cd data/
$ mkdir input
$ mkdir output
$ mkdir output/frames
$ mkdir saved
```
The result videos will be in the 'output' folder

### Example

<div style="text-align:center" markdown="1">

![alt text](https://github.com/SyPRX/Video-decorruption/blob/master/images/corrupt_data.gif)
![alt text](https://github.com/SyPRX/Video-decorruption/blob/master/images/decorrupt_data.gif)

</div>

## Usage

####  One shot decorruption

```bash
python3 main.py --decorruption video_path
```

Where `video_path` is the actual video file path

Example: ``` python3 main.py --decorruption data/input/corrupted_video.mp4``` 

#### Step by step:

* Decompose the video into individual frames:

  ​	``` python3 main.py --deconstruct video_path``` 

* Extract features:

  ​	``` python3 main.py --get_features```

* Reconstruct the sequence:

  ​	```python3 main.py --reduce_dim```