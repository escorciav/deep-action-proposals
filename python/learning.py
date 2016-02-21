import argparse
import json
import logging
import os
import time
from argparse import ArgumentDefaultsHelpFormatter

import hickle as hkl
import lasagne
import numpy as np
import theano
import theano.tensor as T
from sklearn.metrics import average_precision_score, roc_auc_score

from model import weigthed_binary_crossentropy
from model import build_model, read_model
from utils import balance_labels


# ################# Load toy-example of Thumos14 dataset ######################
def load_dataset(prefix, suffix):
    filename = os.path.join(prefix, 'train_fc7_{}.hkl'.format(suffix))
    X_train = hkl.load(filename).astype(np.float32)
    filename = os.path.join(prefix, 'train_conf.hkl')
    y_train = hkl.load(filename).astype(np.uint8)

    filename = os.path.join(prefix, 'val_fc7_{}.hkl'.format(suffix))
    X_val = hkl.load(filename).astype(np.float32)
    filename = os.path.join(prefix, 'val_conf.hkl')
    y_val = hkl.load(filename).astype(np.uint8)

    filename = os.path.join(prefix, 'train_priors.hkl')
    priors = hkl.load(filename).astype(np.float32).flatten()
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return priors, X_train, y_train, X_val, y_val


# ############################# Batch iterator ################################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# #############################################################################

def dump_hyperprm(prmfile, exp_id, model, num_epochs, alpha, batch_size,
                  l_rate, grad_clip, rng_seed, init_model, output_dir, val_ap,
                  rec_50):
    logging.info("Serializing hyper-parameters ...")
    with open(prmfile, 'w') as f:
        json.dump({'exp_id': exp_id, 'model': model, 'num_epochs': num_epochs,
                   'alpha': alpha, 'batch_size': batch_size, 'l_rate': l_rate,
                   'rng_seed': rng_seed, 'init_model': init_model,
                   'grad_clip': grad_clip, 'output_dir': output_dir,
                   'val_ap': float(val_ap), 'rec_50': float(rec_50)},
                  f, indent=4, separators=(',', ': '))
    logging.info("Hpyer-parameters saved on " + prmfile)


def dump_model(filename, network):
    logging.info("Serializing model ...")
    np.savez(filename, *lasagne.layers.get_all_param_values(network))
    logging.info("Model saved on " + filename)


def report_metrics(y_val, val_conf, batch_size):
        n = (y_val.shape[0] / batch_size) * batch_size
        y_true = y_val[0:n, :].flatten()
        y_pred = np.vstack(val_conf).flatten()

        val_ap = average_precision_score(y_true, y_pred)
        val_roc = roc_auc_score(y_true, y_pred)

        n_pos = y_true.sum()
        idx_sorted = np.argsort(-y_pred)
        val_rec = []

        logging.info("Val-AP {:.6f}".format(val_ap))
        logging.info("Val-ROC {:.6f}".format(val_roc))
        for i, v in enumerate([10, 25, 50, 75, 90, 100]):
            tp = y_true[idx_sorted[:int(v * n / 100)]].sum()
            val_rec.append(tp * 1.0 / n_pos)
            logging.info("Val-R{} {:.6f}".format(v, val_rec[i]))
        return val_ap, val_rec[2]


# ############################## Main program #################################

def main(exp_id='0', model='', num_epochs=500, alpha=0.3, batch_size=500,
         l_rate=0.01, grad_clip=100, rng_seed=None, init_model=None,
         output_dir='', ds_prefix=None, ds_suffix=None, snapshot_freq=125,
         **kwargs):
    # Setup logging
    output_dir = os.path.join(output_dir, exp_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logfile = os.path.join(output_dir, exp_id + '.log')
    logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO,
                        format='%(asctime)s %(message)s')

    # Load the dataset
    logging.info("Loading data")
    priors, X_train, y_train, X_val, y_val = load_dataset(ds_prefix, ds_suffix)
    feat_dim = X_train.shape[-1]
    w1_train, w0_train = balance_labels(y_train, batch_size)
    w1_val, w0_val = balance_labels(y_val, batch_size)
    logging.info("Data loaded successfully")

    # Prepare Theano variables for inputs and targets
    if X_train.ndim == 2:
        input_var = T.matrix('inputs')
    elif X_train.ndim == 3:
        input_var = T.tensor3('inputs')
    else:
        msg = 'Unexpected n-dim for X_train: {}'
        raise ValueError(msg.format(X_train.ndim))
    target_loc_var = T.vector('targets_loc')
    target_conf_var = T.imatrix('targets_conf')
    w1_train_var, w0_train_var = T.constant(w1_train), T.constant(w0_train)
    w1_val_var, w0_val_var = T.constant(w1_val), T.constant(w0_val)

    # Create neural network model
    logging.info("Building model and compiling functions...")
    network = build_model(model, input_var, input_size=feat_dim,
                          grad_clip=grad_clip)

    # Loss expression for training, i.e., a scalar objective we want to
    # minimize:
    loc, conf = lasagne.layers.get_output(network)
    loss_match = lasagne.objectives.squared_error(loc, target_loc_var)
    loss_conf = weigthed_binary_crossentropy(conf, target_conf_var,
                                             w0_train_var, w1_train_var)
    loss = alpha * loss_match.mean() + loss_conf.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=l_rate, momentum=0.9)

    # The crucial difference here is that we do a deterministic forward pass
    # through the network, disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loc, test_conf = test_prediction
    test_loss_match = lasagne.objectives.squared_error(test_loc,
                                                       target_loc_var)
    test_loss_conf = weigthed_binary_crossentropy(test_conf, target_conf_var,
                                                  w0_val_var, w1_val_var)
    test_loss = alpha * test_loss_match.mean() + test_loss_conf.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_loc_var, target_conf_var],
                               loss, updates=updates,
                               allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_loc_var, target_conf_var],
                             [test_loss, test_conf],
                             allow_input_downcast=True)

    # Initialize model from previous file
    if init_model and len(init_model) == 2:
        filename, epoch_0 = init_model
        if os.path.exists(filename):
            read_model(filename, network)
            msg = "model {} loaded succesfully"
            epoch_0 = int(epoch_0)
            num_epochs = num_epochs - epoch_0
        else:
            epoch_0 = 0
            msg = "model {} does not exist so training form scratch."
        logging.info(msg.format(filename))
    else:
        epoch_0 = 0

    # Initial hyper-parameters values
    prmfile = os.path.join(output_dir, 'hyper_prm.json')
    dump_hyperprm(prmfile, exp_id, model, num_epochs, alpha, batch_size,
                  l_rate, grad_clip, rng_seed, init_model, output_dir, 0, 0)

    # Finally, launch the training loop.
    logging.info("Starting training...")
    # We iterate over epochs:
    for epoch in xrange(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size,
                                         shuffle=False):
            inputs, targets = batch
            # priors can be a T.constants vector
            train_err += train_fn(inputs, priors, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err, val_conf = 0, []
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size,
                                         shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, priors, targets)
            val_err += err[0]
            val_conf.append(err[1])
            val_batches += 1

        # Then we print the results for this epoch
        logging.info("Epoch {}".format(epoch_0 + epoch + 1))
        logging.info("Elapsed time {:.3f}".format(time.time() - start_time))
        logging.info("Train-loss {:.6f}".format(train_err / train_batches))
        logging.info("Val-loss {:.6f}".format(val_err / val_batches))
        val_ap, rec50 = report_metrics(y_val, val_conf, batch_size)

        # Snapshot of the model
        if (epoch + 1) % snapshot_freq == 0:
            dump_model(os.path.join(output_dir,
                                    'model-{}.npz'.format(epoch + 1)),
                       network)
    logging.info("Training done!!!")

    # Dump the network
    modelfile = os.path.join(output_dir, 'model.npz')
    dump_model(modelfile, network)
    prmfile = os.path.join(output_dir, 'hyper_prm.json')
    dump_hyperprm(prmfile, exp_id, model, num_epochs, alpha, batch_size,
                  l_rate, grad_clip, rng_seed, init_model, output_dir, val_ap,
                  rec50)


def input_parser():
    description = "Train MLP/LSTM using Lasagne."
    p = argparse.ArgumentParser(description=description,
                                formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('-id', '--exp_id', help='Experiment ID')
    h_model = ("'mlp:OUT,DEPTH,WIDTH,DROP_IN,DROP_HID' with DEPTH hidden"
               "layers of WIDTH units, DROP_IN input dropout and DROP_HID"
               "hidden dropout, OUT number of priors."
               "'lstm:OUT,SEQ-LENGTH,WIDTH,DEPTH'. "
               "OUT number of proposals on SEQ-LENGTH. SEQ-LENGTH is the "
               "clip length. WIDTH are the number of units in the LSTM. DEPTH "
               "is the number of LSTM layers.")
    p.add_argument('-m', '--model', help=h_model)
    h_alpha = 'trade-off between matching and confidence loss'
    p.add_argument('-a', '--alpha', help=h_alpha, default=0.3, type=float)
    h_epochs = 'number of training epochs to perform'
    p.add_argument('-ne', '--num_epochs', help=h_epochs, default=500, type=int)
    h_lrate = 'Learning rate'
    p.add_argument('-lr', '--l_rate', help=h_lrate, default=0.01, type=float)
    h_gradclip = 'Gradient clipping'
    p.add_argument('-gc', '--grad_clip', help=h_gradclip, default=100,
                   type=float)
    h_rngseed = 'Seed for random number generation'
    p.add_argument('-rng', '--rng_seed', help=h_rngseed, default=None,
                   type=int)
    h_initmodel = ('Pair of model-path, epoch to restart learning from this '
                   'point')
    p.add_argument('-i', '--init_model', nargs=2, default=None,
                   help=h_initmodel)
    h_dsprefix = 'Fullpath prefix for train/val dataset'
    p.add_argument('-sf', '--snapshot_freq', default=150, type=int,
                   help='Frequency of snapshots')
    dflt_dsprefix = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'data',
        'experiments', 'thumos14', 'a01')
    p.add_argument('-dp', '--ds_prefix', help=h_dsprefix, type=str,
                   default=dflt_dsprefix)
    h_dssuffix = 'Suffix used to read features to train/val model'
    p.add_argument('-ds', '--ds_suffix', help=h_dssuffix, type=str,
                   default='raw')
    h_outputdir = 'Fullpath of folder to save model'
    p.add_argument('-od', '--output_dir', help=h_outputdir, default='')
    return p


if __name__ == '__main__':
    p = input_parser()
    args = vars(p.parse_args())
    main(**args)
