from subprocess import check_call
import os
import csv
import numpy as np

# The data sets to be downloaded
d_sets = [
  'yt_bb_detection_train',
  'yt_bb_detection_validation'
]

# Column names for detection CSV files
col_names = ['youtube_id', 'timestamp_ms','class_id','class_name',
             'object_id','object_presence','xmin','xmax','ymin','ymax']

# Host location of segment lists
web_host = 'https://research.google.com/youtube-bb/'


# Video clip class
class video_clip(object):
  def __init__(self,
               name,
               yt_id,
               timestamp,
               presence,
               bbox,
               class_id,
               obj_id,
               d_set_dir):
    # name = yt_id+class_id+object_id
    self.name     = name
    self.yt_id    = yt_id
    self.timestamps = [timestamp]
    self.presences = [presence]
    self.bboxes = [bbox]
    self.class_id = class_id
    self.obj_id   = obj_id
    self.d_set_dir = d_set_dir

  def print_all(self):
    print('['+self.name+', '+ \
          self.yt_id+', '+ \
          self.timestamps+', '+ \
          self.presences+', '+ \
          self.bboxes+', '+ \
          self.class_id+', '+ \
          self.obj_id+']\n')

# Video class
class video(object):
  def __init__(self,yt_id,first_clip):
    self.yt_id = yt_id
    self.clips = [first_clip]
  def print_all(self):
    print(self.yt_id)
    for clip in self.clips:
      clip.print_all()


# Help function to get the index of the element in an array the nearest to a value
def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


# Parse the annotation csv file and schedule downloads and cuts
def parse_annotations(d_set,dl_dir):

  d_set_dir = dl_dir+'/'+d_set

  # Download & extract the annotation list
  if not os.path.exists(d_set+'.csv'):
    print (d_set+': Downloading annotations...')
    check_call(' '.join(['wget', web_host+d_set+'.csv.gz']),shell=True)
    print (d_set+': Unzipping annotations...')
    check_call(' '.join(['gzip', '-d', '-f', d_set+'.csv.gz']), shell=True)

  print (d_set+': Parsing annotations into clip data...')

  # Parse csv data.
  annotations = []
  with open((d_set+'.csv'), 'rt') as f:
    reader = csv.reader(f)
    annotations = list(reader)

  # Sort to de-interleave the annotations for easier parsing. We use
  # `int(l[1])` to sort by the timestamps numerically; the other fields are
  # sorted lexicographically as strings.
  print(d_set + ': Sorting annotations...')

  # Sort by youtube_id, class, obj_id and then timestamp
  annotations.sort(key=lambda l: (l[0], l[2], l[4], int(l[1])))

  current_clip_name = ['blank']
  clips             = []

  # Parse annotations into list of clips with names, youtube ids, start
  # times and stop times
  for idx, annotation in enumerate(annotations):
    yt_id    = annotation[0]
    timestamp = int(annotation[1])
    class_id = annotation[2]
    obj_id   = annotation[4]
    presence = annotation[5]
    bbox = [float(annotation[6]), float(annotation[7]), float(annotation[8]), float(annotation[9])]

    clip_name = yt_id+'+'+class_id+'+'+obj_id

    # If this is a new clip
    if clip_name != current_clip_name:

      #if idx != 0:
      #  print(clips[-1].name, clips[-1].timestamps)

      # Add the starting clip
      clips.append( video_clip(
        clip_name,
        yt_id,
        timestamp,
        presence,
        bbox,
        class_id,
        obj_id,
        d_set_dir) )

      # Update the current clip name
      current_clip_name = clip_name

    else:
      clips[-1].timestamps.append(timestamp)
      clips[-1].presences.append(presence)
      clips[-1].bboxes.append(bbox)

  # Sort the clips by youtube id
  clips.sort(key=lambda x: x.yt_id)

  # Create list of videos to download (possibility of multiple clips
  # from one video)
  current_vid_id = ['blank']
  vids = []
  for clip in clips:

    vid_id = clip.yt_id

    # If this is a new video
    if vid_id != current_vid_id:
      # Add the new video with its first clip
      vids.append( video ( \
        clip.yt_id, \
        clip ) )
    # If this is a new clip for the same video
    else:
      # Add the new clip to the video
      vids[-1].clips.append(clip)

    # Update the current video name
    current_vid_id = vid_id

  return annotations,clips,vids
