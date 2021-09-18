# Preprocessing Youtube BoundingBox 

Link: https://research.google.com/youtube-bb/download.html

## Create symbolic link
```shell
ln -sbf $PWD/youtube_bb ./dataset
```
**note**: `$PWD/youtube_bb` is a directory to store dataset. Suppose you have sufficient space under `$PWD/youtube_bb`

## Get curated dataset

### Method1: curate from original videos

#### Step1: Download video and extract annotated frames (~400GB)
```shell
python download.py --dl_dir=./youtube_bb --num_threads=16
```

`--data_type`: choose the data type (0: train, default, 1: validation)

#### Trouble Shooting
After downloading more than 50000 videos, you may facing the 429 Error in command of `youtube-dl` like follows:
```
$ youtube-dl  --verbose  -o ./test.mp4 http://youtu.be/Gtmo9JVDSEk
[debug] System config: []
[debug] User config: []
[debug] Custom config: []
[debug] Command-line args: ['--verbose', '--cookies', 'cookies.txt', '-o', './test.mp4', 'http://youtu.be/Gtmo9JVDSEk']
[debug] Encodings: locale UTF-8, fs utf-8, out UTF-8, pref UTF-8
[debug] youtube-dl version 2020.09.20
[debug] Python version 3.5.2 (CPython) - Linux-4.4.0-189-generic-x86_64-with-Ubuntu-16.04-xenial
[debug] exe versions: none
[debug] Proxy map: {}
[youtube] Gtmo9JVDSEk: Downloading webpage
ERROR: Unable to download webpage: HTTP Error 429: Too Many Requests (caused by <HTTPError 429: 'Too Many Requests'>); please report this issue on https://yt-dl.org/bug . Make sure you are using the latest version; see  https://yt-dl.org/update  on how to update. Be sure to call youtube-dl with the --verbose flag and include its complete output.
```

What you should do is to pass the cookies.txt to `--cookies` for youtube-dl:
```
youtube-dl  -o ./test.mp4 http://youtu.be/Gtmo9JVDSEk  --verbose --cookies youtube.com_cookies.txt
```
Please follow [this instruction](https://www.youtube.com/watch?v=XgnwCQzjau8&has_verified=1&ab_channel=SteveAB4EL) to export cookies.txt from chrome.

Then, please try following command
```
python download.py --dl_dir=./youtube_bb --num_threads=16 --vid_start=54900 --cookies_file youtube.com_cookies.txt
```
`--vid_start`: is the offset of starting video to skip the downloaded vidoes

**note**: if you halt the process, `youtube.com_cookies.txt` might loss the content.


#### Step2: Crop & Generate data info (~ 1day?)
```shell
python curate_video.py
```
**note1**: you need to also curate validation data by: `python curate.py --num_threads=2`

**note2**: if you resume from the middle, please try: `python curate.py --num_threads=2 --vid_start=50600`


### Method2: download curated dataset from Google Drive
Because the original videos in youtube are very difficult to download, we provide the curated dataset with following URL:
https://drive.google.com/file/d/1Vuvho89c1yEZbJzmgtc3qnfVIctg4zNt/view?usp=sharing

Please download this under `./dataset`(`${HOME}/trtr/datasets/yt_bb/dataset`) and extract it. Then you can get a directory called `Curation`
