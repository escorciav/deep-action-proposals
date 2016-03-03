import lasagne
import numpy as np
import theano.tensor as T

EPSILON = 10e-8


def build_lstm(input_var=None, seq_length=256, depth=2, width=512,
               input_size=4096, grad_clip=100, forget_bias=5.0):
    """Create LSTM with `depth` number of hidden layers of size `units`
    """
    network = lasagne.layers.InputLayer(shape=(None, seq_length, input_size),
                                        input_var=input_var)

    # Hidden layers
    nonlin = lasagne.nonlinearities.tanh
    gate = lasagne.layers.Gate
    for _ in range(depth):
        network = lasagne.layers.LSTMLayer(
            network, width, grad_clipping=grad_clip, nonlinearity=nonlin,
            forgetgate=gate(b=lasagne.init.Constant(forget_bias)))

    # Retain last-output state
    network = lasagne.layers.SliceLayer(network, -1, 1)
    return network


def build_mlp(input_var=None, depth=2, width=1024, drop_input=.2,
              drop_hidden=.5, input_size=4096):
    """Create an MLP with `depth` number of hidden layers of size `width`
    """
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, input_size),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)

    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)

    return network


def build_model(model_prm=None, input_var=None, input_size=4096,
                grad_clip=100, forget_bias=1.0):
    """Create localization model
    """
    if model_prm.startswith('mlp:'):
        user_prm = model_prm.split(':', 1)[1].split(',')
        n_outputs, depth, width, drop_in, drop_hid = user_prm
        network = build_mlp(input_var, int(depth), int(width), float(drop_in),
                            input_size=input_size)
    elif model_prm.startswith('lstm:'):
        user_prm = model_prm.split(':', 1)[1].split(',')
        n_outputs, seq_length, width, depth = user_prm
        network = build_lstm(input_var, int(seq_length), int(depth),
                             int(width), input_size=input_size,
                             grad_clip=grad_clip, forget_bias=forget_bias)
    else:
        raise ValueError("Unrecognized model type " + model_prm)

    # Output layer
    nonlin, n_outputs = lasagne.nonlinearities.sigmoid, int(n_outputs)
    localization = lasagne.layers.DenseLayer(network, n_outputs * 2)
    conf = lasagne.layers.DenseLayer(network, n_outputs, nonlinearity=nonlin)
    return localization, conf


def read_model(filename, network):
    """Set parameters of lasagne network from a file

    Parameters
    ----------
    filename : str
        Fullpath of npz-file with weights of the network
    network : lasagne expresion
        Network built with lasagne

    """
    with np.load(filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    return None


def weigthed_binary_crossentropy(predictions, targets, w0, w1):
    """Computes the binary weigthed cross entropy loss between predictions and
    targets.
    Parameters
    ----------
    predictions : Theano tensor
        Predictions in (0, 1), such as sigmoidal output of a neural network.
    targets : Theano tensor
        Targets in {0, 1}.
    w0 : Theano constant
        Weight for class 0.
    w1 : Theano constant
        Weight for class 1.
    Returns
    -------
    Theano tensor
        An expression for the element-wise binary hinge loss
    TODO: Match shape of predictions and targets.
    """
    pos_log = T.log(T.clip(predictions, EPSILON, np.inf))
    neg_log = T.log(T.clip(1.0 - predictions, EPSILON, np.inf))
    return -(w1 * targets * pos_log + w0 * (1.0 - targets) * neg_log)
