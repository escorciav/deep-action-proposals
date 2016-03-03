import difflib
import glob
import json
import os

import pandas as pd
import numpy as np

from utils import levenshtein_distance

ACTIVITYNET_ANNOTATION_FILE = 'activity_net.v1-2.gt.json'
OVERLAPPED_CATEGORY_IDS = [159, 82, 233, 224, 195, 116, 80, 106, 169]

class ActivityNet(object):
    fields_video = ['video-name', 'duration', 'frame-rate', 'n-frames']
    fields_segment = ['video-name', 't-init', 't-end', 'f-init', 'n-frames',
                      'video-duration', 'frame-rate', 'video-frames',
                      'label-idx']

    def __init__(self, dirname='data/activitynet',
                 annotation_file=ACTIVITYNET_ANNOTATION_FILE,
                 overlapped_category_ids=OVERLAPPED_CATEGORY_IDS):
        """Initialize thumos14 class

        Parameters
        ----------
        dirname : string
            Fullpath of folder with THUMOS-14 data

        """
        if not os.path.isdir(dirname):
            raise IOError('Unexistent directory {}'.format(dirname))
        self.root = dirname
        self.info = os.path.join(dirname, 'info')
        self.annotation_filename = os.path.join(self.info, annotation_file)
        self.overlapped = overlapped_category_ids
        
        # Read index used on ActivityNet
        self.index_filename = os.path.join(self.info,
                                           'class_index_detection.txt')
        
        # Video CSV
        self.files_video_list = [
            os.path.join(self.root, 'metadata', 'train_list.txt'),
            os.path.join(self.root, 'metadata', 'val_list.txt'),
            os.path.join(self.root, 'metadata', 'test_list.txt')]
        msg = 'Unexistent list of {} videos and its information'
        if not os.path.isfile(self.files_video_list[0]):
            try:
                self._gen_video_list(self.files_video_list[0], 'train')
            except:
                raise IOError(msg.format('training'))
        if not os.path.isfile(self.files_video_list[1]):
            try:
                self._gen_video_list(self.files_video_list[1], 'val')
            except:
                raise IOError(msg.format('validation'))
        if not os.path.isfile(self.files_video_list[2]):
            try:
                self._gen_video_list(self.files_video_list[2], 'test')
            except:
                raise IOError(msg.format('testing'))

        # Segments CSV
        self.files_seg_list = [
            os.path.join(self.root, 'metadata', 'train_segments_list.txt'),
            os.path.join(self.root, 'metadata', 'val_segments_list.txt'),
            os.path.join(self.root, 'metadata', 'test_segments_list.txt')]
        if not os.path.isfile(self.files_seg_list[0]):
            self._gen_segments_info(self.files_seg_list[0], 'train')
        if not os.path.isfile(self.files_seg_list[1]):
            self._gen_segments_info(self.files_seg_list[1], 'val')
        if not os.path.isfile(self.files_seg_list[2]):
            self._gen_segments_info(self.files_seg_list[2], 'test')

    def _gen_video_list(self, filename, set_choice='train'):
        """Create CSV with information about ActivityNet videos

        Parameters
        ----------
        filename : str
            Fullpath of CSV-file
        set_choice : str
            ('train','val' or 'test') dump annotations of the corresponding set

        """
        video_info_filename = os.path.join(self.info, 
                                           '{}.txt'.format(set_choice))
        video_list = np.array(pd.read_csv(video_info_filename, 
                                          header=None)).flatten()
        with open(self.annotation_filename, 'r') as fobj:
            data = json.load(fobj)['database']
        v_noex_lst, dur_lst, n_frames_lst, frame_rate_lst = [], [], [], []
        for v in video_list:
            # Get duration from raw annotations.
            v_noex = os.path.splitext(v)[0]
            dur = data[v_noex[-11:]]['duration'] # Excluding v_ chars.
            # Get number of frames from extracted frames count.
            frm_dir = os.path.join(self.root, 
                                   'frm/{}/{}'.format(set_choice, v_noex))
            n_frames = len(glob.glob(os.path.join(frm_dir, '*.jpg')))
            # Frame rate computed from dur and number of frames.
            frame_rate = (n_frames * 1.0) / dur
            dur_lst.append(dur)
            n_frames_lst.append(n_frames)
            frame_rate_lst.append(frame_rate)
            v_noex_lst.append(v_noex)
        df = pd.DataFrame({'video-name': v_noex_lst,
                           'duration': dur_lst,
                           'frame-rate': frame_rate_lst,
                           'n-frames': n_frames_lst})
        if not os.path.isdir(os.path.join(self.root, 'metadata')):
            os.makedirs(os.path.join(self.root, 'metadata'))
        output_file = os.path.join(self.root, 'metadata', 
                                   '{}_list.txt'.format(set_choice))
        df.to_csv(output_file, sep=' ', index=False, 
                  header=True, columns=self.fields_video)
        return df

    def _gen_segments_info(self, filename, set_choice, id_prepend='v_'):
        """Create CSV with information about ActivityNet action segments

        Parameters
        ----------
        filename : str
            Fullpath of CSV-file
        set_choice : str
            ('train','val' or 'test') dump annotations of the corresponding set

        """
        set_choice_helper = {'train': 'training', 'val': 'validation',
                             'test': 'testing'}
        with open(self.annotation_filename, 'r') as fobj:
            data = json.load(fobj)['database']

        # DataFrame fields
        video_name, video_duration, frame_rate, video_frames = [], [], [], []
        t_init, t_end, f_init, n_frames, l_idx = [], [], [], [], []

        # Looking for videos in set choice.
        for v_id, v in data.iteritems():
            if v['subset'] != set_choice_helper[set_choice.lower()]:
                continue
            # Count frames.
            frm_dir = os.path.join(self.root,
                'frm/{}/{}{}'.format(set_choice, id_prepend, v_id))
            video_frames_i = len(glob.glob(os.path.join(frm_dir, '*.jpg')))
            frame_rate_i = (video_frames_i * 1.0) / v['duration']
            # Appending segment info.
            for annotation in v['annotations']:
                video_name.append(id_prepend + v_id)
                video_duration.append(v['duration'])
                frame_rate.append(frame_rate_i)
                video_frames.append(video_frames_i)
                t_init.append(annotation['segment'][0])
                t_end.append(annotation['segment'][1])
                f_i = np.floor(annotation['segment'][0] * frame_rate_i)
                f_init.append(f_i)
                f_e = np.floor(annotation['segment'][1] * frame_rate_i)
                n_frames.append(f_e - f_i + 1.0)
                l_idx.append(self.index_from_action_name(annotation['label']))
        
        # Build DataFrame.
        df = pd.DataFrame({'video-name': video_name, 't-init': t_init, 
                           't-end': t_end, 'f-init': f_init,
                           'n-frames': n_frames, 
                           'video-duration': video_duration,
                           'frame-rate': frame_rate,
                           'video-frames': video_frames,
                           'label-idx': l_idx})
        if isinstance(filename, str):
            df.to_csv(filename, sep=' ', index=False, columns=self.fields_segment)
        return df

    def dir_videos(self, set_choice='train'):
        """Return string of folder of annotations

        Parameters
        ----------
        set_choice : string, optional
            ('train', 'val' or 'test') set of interest

        """
        set_choice = set_choice.lower()
        if set_choice == 'val' or set_choice == 'validation':
            return os.path.join(self.root, 'val_videos')
        elif (set_choice == 'test' or set_choice == 'testing' or
              set_choice == 'tst'):
            return os.path.join(self.root, 'test_videos')
        elif (set_choice == 'train' or set_choice == 'training' or
              set_choice == 'trng'):
            return os.path.join(self.root, 'train_videos')
        else:
            raise ValueError('unrecognized choice')

    def index_from_action_name(self, name):
        df = pd.read_csv(self.index_filename)
        idx = df['action-name'] == name
        return int(df.loc[idx, 'index'])

    def segments_info(self, set_choice='train', filename=None):
        """Return a DataFrame with information about action segments

        Parameters
        ----------
        set_choice : string, optional
            ('train','val' or 'test') dump annotations of the corresponding set

        """
        set_choice = set_choice.lower()
        if (set_choice == 'train' or set_choice == 'training' or
            set_choice == 'trng'):
            filename = self.files_seg_list[0]
        elif set_choice == 'val' or set_choice == 'validation':
            filename = self.files_seg_list[1]
        elif (set_choice == 'test' or set_choice == 'testing' or
              set_choice == 'tst'):
            filename = self.files_seg_list[2]
        else:
            raise ValueError('unrecognized choice')

        df = pd.read_csv(filename, sep=' ')
        if df.shape[1] != len(self.fields_segment):
            raise ValueError('Inconsistent number of columns')
        return df

    def video_info(self, set_choice='train'):
        """Return DataFrame with info about videos on the corresponding set

        Parameters
        ----------
        set_choice : string
            ('train', 'val' or 'test') set of interest

        """
        set_choice = set_choice.lower()
        if (set_choice == 'train' or set_choice == 'training' or
            set_choice == 'trng'):
            filename = self.files_video_list[0]
        elif set_choice == 'val' or set_choice == 'validation':
            filename = self.files_video_list[1]
        elif (set_choice == 'test' or set_choice == 'testing' or
              set_choice == 'tst'):
            filename = self.files_video_list[2]
        else:
            raise ValueError('unrecognized choice')

        df = pd.read_csv(filename, sep=' ')
        if df.shape[1] != len(self.fields_video):
            raise ValueError('Inconsistent number of columns')
        return df

    def get_segments_from_overlapped_categories(self, df):
        return df[df['label-idx'].isin(self.overlapped).copy()]
