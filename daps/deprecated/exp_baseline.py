import numpy as np
import pandas as pd

from baseline import BaselineData, TempPriorsNoScale
from baseline import proposals_per_video

from utilities import dump_json
from utilities import segment_format

NUM_PROPOSALS = [1, 10] + range(100, 1000, 100) + range(1000, 10001, 1000)


def eval_temporal_priors(train_file, test_file, n_prop=NUM_PROPOSALS,
                         filename=None):
    """Run TempPriorsNoScale over a range of number-of-proposals
    """
    ds_train = BaselineData.fromcsv(train_file)
    Xtrain = ds_train.get_temporal_loc()
    ds_test_df = pd.read_csv(test_file, sep=' ')
    Ztest = np.array(ds_test_df.loc[:, 'n-frames'])

    for i, v in enumerate(n_prop):
        if v > Xtrain.shape[0]:
            # Use all annotations as priors ;)
            continue

        m = TempPriorsNoScale(v)
        m.fit(Xtrain)
        Ypred_centered, idx = m.proposals(Ztest, return_index=True)

        Ypred = segment_format(Ypred_centered, 'c2b')
        # Form video-proposals format [f-init, f-end, score]
        vid_prop_all = np.hstack([Ypred, np.zeros((Ypred.shape[0], 1))])
        vid_prop = proposals_per_video(vid_prop_all, v)
        id_prop = dict(zip(ds_test_df.loc[:, 'video-name'].tolist(),
                           vid_prop.tolist()))

        if isinstance(filename, str):
            idfile = filename + '.n-prop_{}'.format(v)
            dump_json(idfile, id_prop)
    return None
