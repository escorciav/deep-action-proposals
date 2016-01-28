import difflib
import glob
import os

import pandas as pd
import numpy as np

from utils import levenshtein_distance


class Thumos14(object):
    fields_video = ['video-name', 'duration', 'frame-rate', 'n-frames']
    fields_segment = ['video-name', 't-init', 't-end', 'f-init', 'n-frames',
                      'video-duration', 'frame-rate', 'video-frames',
                      'label-idx']

    def __init__(self, dirname='data/thumos14'):
        """Initialize thumos14 class

        Parameters
        ----------
        dirname : string
            Fullpath of folder with THUMOS-14 data

        """
        if not os.path.isdir(dirname):
            raise IOError('Unexistent directory {}'.format(dirname))
        self.root = dirname

        # Read index used on THUMOS-14
        filename = os.path.join(self.root, 'class_index_detection.txt')
        self.df_index_labels = pd.read_csv(filename, header=None, sep=' ')

        # Video CSV
        self.files_video_list = [
            os.path.join(self.root, 'metadata', 'val_list.txt'),
            os.path.join(self.root, 'metadata', 'test_list.txt')]
        msg = 'Unexistent list of {} videos and its information'
        # TODO: Generate list if not exist
        if not os.path.isfile(self.files_video_list[0]):
            raise IOError(msg.format('validation'))
        if not os.path.isfile(self.files_video_list[1]):
            raise IOError(msg.format('testing'))

        # Segments CSV
        self.files_seg_list = [
            os.path.join(self.root, 'metadata', 'val_segments_list.txt'),
            os.path.join(self.root, 'metadata', 'test_segments_list.txt')]
        if not os.path.isfile(self.files_seg_list[0]):
            self._gen_segments_info(self.files_seg_list[0], 'val')
        if not os.path.isfile(self.files_seg_list[1]):
            self._gen_segments_info(self.files_seg_list[1], 'test')

    def annotation_files(self, set_choice='val'):
        """
        Return a list with files of temporal annotations of THUMOS-14 actions

        Parameters
        ----------
        set_choice : string, optional
            ('val' or 'test') set of interest

        """
        dirname = self.dir_annotations(set_choice)
        return glob.glob(os.path.join(dirname, 'annotation', '*.txt'))

    def dir_annotations(self, set_choice='val'):
        """Return string of folder of annotations

        Parameters
        ----------
        set_choice : string, optional
            ('val' or 'test') set of interest

        """
        set_choice = set_choice.lower()
        if set_choice == 'val' or set_choice == 'validation':
            return os.path.join(self.root, 'th14_temporal_annotations_val')
        elif (set_choice == 'test' or set_choice == 'testing' or
              set_choice == 'tst'):
            return os.path.join(self.root, 'th14_temporal_annotations_test')
        else:
            raise ValueError('unrecognized choice')

    def dir_videos(self, set_choice='val'):
        """Return string of folder with videos

        Parameters
        ----------
        set_choice : string, optional
            ('val' or 'test') return folder of the corresponding set

        """
        set_choice = set_choice.lower()
        if set_choice == 'val' or set_choice == 'validation':
            return os.path.join(self.root, 'val_mp4')
        elif (set_choice == 'test' or set_choice == 'testing' or
              set_choice == 'tst'):
            return os.path.join(self.root, 'test_mp4')
        else:
            raise ValueError('unrecognized choice')

    def _gen_segments_info(self, filename, set_choice):
        """Create CSV with information about THUMOS-14 action segments

        Parameters
        ----------
        filename : str
            Fullpath of CSV-file
        set_choice : str
            ('val' or 'test') dump annotations of the corresponding set

        """
        # Read annotations and create labels (0-indexed)
        files = self.annotation_files(set_choice)
        list_df, list_arr = [], []
        for i in files:
            list_df.append(pd.read_csv(i, header=None, sep=' '))
            n = list_df[-1].shape[0]
            list_arr.append(np.ones(n, dtype=int) *
                            self.index_from_filename(i))
        df_s = pd.concat(list_df, ignore_index=True)
        df_l = pd.DataFrame(np.concatenate(list_arr, axis=0),
                            columns=['labels'])

        # Read video list
        df_v = self.video_info(set_choice)

        # Match frame-rate of each segment
        video_id, idx = np.unique(df_s.loc[:, 0], return_inverse=True)
        d = np.zeros((video_id.size, df_v.shape[0]))
        for i, u in enumerate(video_id):
            for j, v in enumerate(df_v.loc[:, 'video-name']):
                d[i, j] = levenshtein_distance(u, v)
        idx_map_vid2seg = d.argmin(axis=1)[idx]
        video_dur = df_v.loc[idx_map_vid2seg, 'duration']
        frame_rate = np.array(df_v.loc[idx_map_vid2seg, 'frame-rate'])
        video_frames = df_v.loc[idx_map_vid2seg, 'n-frames']

        # Compute initial-frame, ending-frame, num-frames
        f_i = np.round(frame_rate * np.array(df_s.loc[:, 2]))
        f_e = np.round(frame_rate * np.array(df_s.loc[:, 3]))
        n_frames = f_e - f_i + 1

        # Create DataFrame
        # df_s[:, 1] is ignored because Thumos14 annotation includes extra
        # blank space.
        # Explicit reindexing is required to avoid Reindexing-Error.
        video_dur.index = video_frames.index = range(video_dur.size)
        df = pd.concat([df_s.loc[:, 0], df_s.loc[:, 2::],
                        pd.DataFrame(f_i), pd.DataFrame(n_frames),
                        video_dur, pd.DataFrame(frame_rate),
                        video_frames, df_l],
                       axis=1, ignore_index=True)
        if isinstance(filename, str):
            df.to_csv(filename, sep=' ', index=False, header=None)
        return df

    def index_from_filename(self, filename):
        """Return index btw [-1, 20) of action inside filename
        """
        basename = os.path.basename(os.path.splitext(filename)[0])
        if 'Ambiguous' in basename:
            return -1
        else:
            match = difflib.get_close_matches(basename,
                                              self.df_index_labels.loc[:, 1])
            return np.where(self.df_index_labels.loc[:, 1] == match[0])[0]

    def segments_info(self, set_choice='val', filename=None):
        """Return a DataFrame with information about THUMOS-14 action segments

        Parameters
        ----------
        set_choice : string, optional
            ('val' or 'test') dump annotations of the corresponding set

        """
        set_choice = set_choice.lower()
        if set_choice == 'val' or set_choice == 'validation':
            filename = self.files_seg_list[0]
        elif (set_choice == 'test' or set_choice == 'testing' or
              set_choice == 'tst'):
            filename = self.files_seg_list[1]
        else:
            raise ValueError('unrecognized choice')

        df = pd.read_csv(filename, header=None, sep=' ')
        if df.shape[1] == len(self.fields_segment):
            df.columns = self.fields_segment
        else:
            raise ValueError('Inconsistent number of columns')
        return df

    def video_info(self, set_choice='val'):
        """Return DataFrame with info about videos on the corresponding set

        Parameters
        ----------
        set_choice : string
            ('val' or 'test') set of interest

        """
        set_choice = set_choice.lower()
        if set_choice == 'val' or set_choice == 'validation':
            filename = self.files_video_list[0]
        elif (set_choice == 'test' or set_choice == 'testing' or
              set_choice == 'tst'):
            filename = self.files_video_list[1]
        else:
            raise ValueError('unrecognized choice')

        df = pd.read_csv(filename, header=None, sep=' ')
        if df.shape[1] == len(self.fields_video):
            df.columns = self.fields_video
        else:
            raise ValueError('Inconsistent number of columns')
        return df
