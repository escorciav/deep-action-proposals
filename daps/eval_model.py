import lasagne
import numpy as np

from daps.c3d_encoder import Feature
from daps.utils.segment import format as segment_format


def forward_pass(network, input_data):
    """Forward pass input_data over network
    """
    l_pred_var, y_pred_var = lasagne.layers.get_output(network, input_data,
                                                       deterministic=True)
    loc = l_pred_var.eval().reshape((-1, 2))
    return loc, y_pred_var.eval()


def nms_detections(dets, score, overlap=0.3):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    dets : ndarray.
        Each row is ['f-init', 'f-end']
    score : 1darray.
        Detection score.
    overlap : float.
        Minimum overlap ratio (0.3 default).

    Outputs
    -------
    dets : ndarray.
        Remaining after suppression.
    """
    t1 = dets[:, 0]
    t2 = dets[:, 1]
    ind = np.argsort(score)

    area = (t2 - t1 + 1).astype(float)

    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]

        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])

        wh = np.maximum(0., tt2 - tt1 + 1.0)
        o = wh / (area[i] + area[ind] - wh)

        ind = ind[np.nonzero(o <= overlap)[0]]

    return dets[pick, :], score[pick]


def retrieve_proposals(video_name, l_size, network, T=256, stride=128,
                       c3d_size=16, c3d_stride=8, pool_type='mean',
                       hdf5_dataset=None, model_prm=None):
    """Retrieve proposals for an input video.

    Parameters
    ----------
    video_name : str.
        Video identifier.
    l_size : int.
        Size of the video.
    network : (localization, conf).
        Lasagne layers.
    T : int, optional.
        Canonical temporal size of evaluation window.
    stride : int, optional.
        Size of the sliding step.
    c3d_size : int, optional.
        Size of temporal fiel C3D network.
    c3d_stride : int, optional.
        Size of temporal stride between extracted features.
    pool_type : str, optional.
        Global pooling strategy over a bunch of features.
        'mean', 'max', 'pyr-2-mean/max', 'concat-2-mean/max'
    hdf5_dataset : str.
        Path to feature file.

    """
    # IO interface.
    fobj = Feature(filename=hdf5_dataset, t_size=c3d_size,
                   t_stride=c3d_stride, pool_type=pool_type)
    fobj.open_instance()
    # Video scanning.
    f_init_array = np.arange(0, l_size - T, stride)
    feat_stack = fobj.read_feat_batch_from_video(video_name, f_init_array,
                                                 duration=T).astype(np.float32)
    if model_prm.startswith('lstm:'):
        user_prm = model_prm.split(':', 1)[1].split(',')
        n_outputs, seq_length, width, depth = user_prm
        feat_stack = feat_stack.reshape(feat_stack.shape[0],
                                        int(seq_length),
                                        feat_stack.shape[1]/int(seq_length))

    # Close instance.
    fobj.close_instance()

    # Generate proposals.
    loc, score = forward_pass(network, feat_stack)
    n_proposals = score.shape[1]
    n_segments = score.shape[0]
    score = score.flatten()
    map_array = np.stack((f_init_array,
                          np.zeros(n_segments))).repeat(n_proposals, axis=-1).T
    proposal = segment_format(map_array + (loc.clip(0, 1) * T),
                              'c2b').astype(int)
    return proposal, score
