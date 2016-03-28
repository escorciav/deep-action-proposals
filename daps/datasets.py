import difflib
import glob
import json
import os

import numpy as np
import pandas as pd

from utilities import levenshtein_distance

ACTIVITYNET_ANNOTATION_FILE = 'activity_net.v1-2.gt.json'
ANET_SIMILAR_CLASS_IDS_WITH_THUMOS14 = [159, 82, 233, 224, 195,
                                        116, 80, 106, 169]


class Dataset(object):
    """Wrapper around classes packing dataset information

    Attributes
    ----------
    wrapped_dataset : DatasetBase
        wrapped dataset

    ToDo
    -----
    Create a super class for Thumos14

    """
    def __init__(self, name, **kwargs):
        """Setup dataset object

        Parameters
        ----------
        name : str
            Name of dataset to use

        """
        if type(name) is not str:
            raise ValueError('name must be of type str')
        name = name.lower()

        if name == 'thumos14' or name == 'thumos_14':
            self.wrapped_dataset = Thumos14(**kwargs)
        elif name == 'activitynet':
            self.wrapped_dataset = ActivityNet(**kwargs)
        else:
            raise ValueError('Unknown dataset {}'.format(name))

    def __getattr__(self, attr):
        orig_attr = self.wrapped_dataset.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if isinstance(result, type(self.wrapped_dataset)):
                    if result == self.wrapped_dataset:
                        return self
                return result
            return hooked
        else:
            return orig_attr


class DatasetBase(object):
    """Primitive class to pack information about dataset
    """
    msg_overload = 'This method should be overloaded'
    fields_video = ['video-name', 'duration', 'frame-rate', 'n-frames']
    fields_segment = ['video-name', 't-init', 't-end', 'f-init', 'n-frames',
                      'video-duration', 'frame-rate', 'video-frames',
                      'label-idx']

    def segments_info(self):
        raise NotImplemented(self.msg_overload)

    def video_info(self):
        raise NotImplemented(self.msg_overload)


class ActivityNet(DatasetBase):
    """Pack data about ActivityNet
    """
    def __init__(self, dirname='data/activitynet',
                 annotation_file=ACTIVITYNET_ANNOTATION_FILE,
                 overlapped_category_ids=ANET_SIMILAR_CLASS_IDS_WITH_THUMOS14):
        """Initialize ActivityNet dataset

        Parameters
        ----------
        dirname : str
            Fullpath of folder with ActivityNet data
        annotation_file : str
            Filename with ground-truth annotation
        overlapped_category_ids : list
            class ids overlapping Thumos14 categories

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
            dur = data[v_noex[-11:]]['duration']  # Excluding v_ chars.
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
            frm_dir = os.path.join(
                self.root, 'frm/{}/{}{}'.format(set_choice, id_prepend, v_id))
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
            df.to_csv(filename, sep=' ', index=False,
                      columns=self.fields_segment)
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
        if set_choice in ['train', 'training', 'trng']:
            filename = self.files_seg_list[0]
        elif set_choice in ['val', 'validation']:
            filename = self.files_seg_list[1]
        elif set_choice in ['test', 'testing', 'tst']:
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
        if set_choice in ['train', 'training', 'trng']:
            filename = self.files_video_list[0]
        elif set_choice in ['val', 'validation']:
            filename = self.files_video_list[1]
        elif set_choice in ['test', 'testing', 'tst']:
            filename = self.files_video_list[2]
        else:
            raise ValueError('unrecognized choice')

        df = pd.read_csv(filename, sep=' ')
        if df.shape[1] != len(self.fields_video):
            raise ValueError('Inconsistent number of columns')
        return df

    def get_segments_from_overlapped_categories(self, df):
        return df[df['label-idx'].isin(self.overlapped).copy()]


class Thumos14(DatasetBase):
    """Pack data about Thumos14 dataset
    """
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
        f_i = np.round(frame_rate * np.array(df_s.loc[:, 2])).clip(1)
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
        df.columns = self.fields_segment
        if isinstance(filename, str):
            df.to_csv(filename, sep=' ', index=False)
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
        if set_choice in ['val', 'validation']:
            filename = self.files_seg_list[0]
        elif set_choice in ['test', 'testing', 'tst']:
            filename = self.files_seg_list[1]
        else:
            raise ValueError('unrecognized choice')

        df = pd.read_csv(filename, sep=' ')
        if df.shape[1] != len(self.fields_segment):
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
        if set_choice in ['val', 'validation']:
            filename = self.files_video_list[0]
        elif set_choice in ['test', 'testing', 'tst']:
            filename = self.files_video_list[1]
        else:
            raise ValueError('unrecognized choice')

        df = pd.read_csv(filename, sep=' ')
        if df.shape[1] != len(self.fields_video):
            raise ValueError('Inconsistent number of columns')
        return df
