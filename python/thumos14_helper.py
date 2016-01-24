import os

import pandas as pd


class thumos14(object):
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

        # TODO: Generate list if not exist
        if not os.path.isfile(os.path.join(self.root, 'metadata',
                                           'val_list.txt')):
            msg = 'Unexistent list of validation videos and its information'
            raise IOError(msg)
        if not os.path.isfile(os.path.join(self.root, 'metadata',
                                           'test_list.txt')):
            msg = 'Unexistent list of testing videos and its information'
            raise IOError(msg)

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
