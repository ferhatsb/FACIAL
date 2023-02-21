# Preprocess

## Downloading Sample Video From YouTube
Using https://github.com/yt-dlp/yt-dlp to download videos from YouTube.

### Get info
```commandline
yt-dlp -F 'https://www.youtube.com/watch?v=X'
```

### Download specific resolution 720p mostly
```commandline
yt-dlp -f '247' 'https://www.youtube.com/watch?v=X'
```

> Clean up video with Davinci etc.

### Get first frame
```commandline
ffmpeg -i sample.webm -vframes 1 1.jpeg
```

### Crop video;
```commandline
ffmpeg -i sample.webm -filter:v "crop=512:512:347:93" crop.mp4
```

### Create images
```commandline
ffmpeg -i train1.mp4 train1_image/%06d.jpg
```

## OpenFace

```commandline
sudo docker run -v $(pwd):/video -it --rm algebr/openface:latest
build/bin/FeatureExtraction -fdir /video/train1_image -out_dir /video
```

### Predictions

Create detections folder under images folder (train1_image) and execute `detections.py`
```commandline
python detections.py --image_folder=~/Development/facial_data/train1_image
```
## DeepFace
```commandline
python test.py --name=model --epoch=20 --img_folder=~/Development/FACIAL/video_preprocess/train1_image
```