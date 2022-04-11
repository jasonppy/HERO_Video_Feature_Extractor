import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import ffmpeg
import math
import csv
import time
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))


class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            csv_fn,
            framerate=1,
            noclip_framerate=1,
            size=112,
            centercrop=False,
            overwrite=False,
            model_version="ViT-B/32",
            scenedetect_folder="/saltpool0/data/pyp/vqhighlight/scenedetect27/"
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv_fn)
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.overwrite = overwrite
        self.model_version = model_version
        self.noclip_framerate = noclip_framerate

        # get a dict of vid:[time_slots]
        self.vid2clips = {}
        for vfn in self.csv['video_path']:
            vid = os.path.basename(vfn)[:-4] # exclude ".mkv"
            temp = vid.split("_")
            start, end = temp[-2], temp[-1]
            suffix = "_" + start + "_" + end
            ytvid = vid[:-len(suffix)]
            start, end = float(start), float(end)
            anuj_vid = ytvid + "_" + str(int(start)) + "_" + str(int(end))
            detect_csv_fn = os.path.join(scenedetect_folder, anuj_vid + "-Scenes.csv")
            ftr = [3600.,60.,1.]
            with open(detect_csv_fn, newline='') as f:
                data = csv.reader(f)
                for row in data:
                    # print(row)
                    if len(row) == 0:
                        self.vid2clips[vid] = None # the scenedetect software couldn't detect scenes
                    else:
                        temp = row[1:] # ['00:00:01.967', '00:00:06.667', '00:00:07.267', '00:00:07.867', ...]
                        self.vid2clips[vid] = [sum([a*b for a,b in zip(ftr, map(int,t_slot.split('.')[0].split(':')))]) + float(t_slot.split('.')[-1])/1000. for t_slot in temp]
                        self.vid2clips[vid].append(float(end - start))
                    break

    def __len__(self):
        return len(self.csv)

    def _get_video_info(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams']
                             if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = math.floor(convert_to_float(video_stream['avg_frame_rate']))
        try:
            frames_length = int(video_stream['nb_frames'])
            duration = float(video_stream['duration'])
        except Exception:
            frames_length, duration = -1, -1
        info = {"duration": duration, "frames_length": frames_length,
                "fps": fps, "height": height, "width": width}
        return info

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)

    def __getitem__(self, idx):
        video_path = self.csv['video_path'].values[idx]
        output_file = self.csv['feature_path'].values[idx]
        if self.model_version == "RN50x4":
            output_file = output_file.replace(
                "clip-vit_features", "clip-rn50x4_features")
        load_flag = os.path.isfile(video_path)
        if not self.overwrite:
            load_flag = load_flag and not(os.path.isfile(output_file))
        if load_flag:
            # print('Decoding video: {}'.format(video_path))
            try:
                info = self._get_video_info(video_path)
                h, w = info["height"], info["width"]
            except Exception:
                print('ffprobe failed at: {}'.format(video_path))
                return {'video': th.zeros(1), 'input': video_path,
                        'output': output_file, 'info': {}}
            height, width = self._get_output_dim(h, w)
            try:
                duration = info["duration"]
                fps = self.framerate
                if duration > 0 and duration < 1/fps+0.1:
                    fps = 2/max(int(duration), 1)
                    print(duration, fps)
            except Exception:
                fps = self.framerate
            all_video = []
            all_len = []
            cmd = (
                    ffmpeg
                    .input(video_path)
                    .filter('fps', fps=1)
                    .filter('scale', width, height)
                    )
            if self.centercrop:
                x = int((width - self.size) / 2.0)
                y = int((height - self.size) / 2.0)
                cmd = cmd.crop(x, y, self.size, self.size)
            s_time = time.time()
            out, _ = (
                cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
            # print(f"ffmpeg time: {time.time() - s_time:.4f}")
            if self.centercrop and isinstance(self.size, int):
                height, width = self.size, self.size
            video = np.frombuffer(out, np.uint8).reshape(
                [-1, height, width, 3])
            video = th.from_numpy(video.astype('float32'))
            video = video.permute(0, 3, 1, 2) # T, C, H, W
            time_slots = self.vid2clips[os.path.basename(video_path)[:-4]]
            # ftr = [3600.,60.,1.]
            if time_slots != None:
                for ii in range(len(time_slots)-1):
                    # start, end = sum([a*b for a,b in zip(ftr, map(int,time_slots[i].split('.')[0].split(':')))]) + float(time_slots[i].split('.')[-1])/1000., sum([a*b for a,b in zip(ftr, map(int,time_slots[i+1].split('.')[0].split(':')))]) + float(time_slots[i+1].split('.')[-1])/1000.
                    # print(start, end)
                    # start, end = int(np.ceil(time_slots[i])), int(np.ceil(time_slots[i+1]))
                    # print()
                    # print(len(video))
                    # print(time_slots)
                    start, end = time_slots[ii], time_slots[ii+1]
                    cur_s = 0
                    for j in range(int(np.ceil(start)), min(int(np.ceil(end)),len(video)), 3):
                        all_video.append(video[j])
                        cur_s += 1
                    if cur_s == 0:
                        all_video.append(th.zeros_like(video[0]))
                    all_len.append(cur_s)
                all_video = th.stack(all_video)
            else:
                all_video = video
                all_len = [len(video)]
        else:
            all_video = th.zeros(1)
            all_len = 1

        return {'video': all_video, 'all_len': all_len, 'input': video_path, 'output': output_file}
