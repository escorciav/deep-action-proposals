# Deep Action Proposals (DAPs) for Videos
Temporal Action Proposals for long untrimmed videos.

DAPs architecture allows to retrieve segments from long videos where it is
likely to find actions with high recall very quickly.

## Citation

If you find any piece of code valuable for your research please cite this work:

```
@Inbook{Escorcia2016,
author="Escorcia, Victor and Caba Heilbron, Fabian and Niebles, Juan Carlos and Ghanem, Bernard",
editor="Leibe, Bastian and Matas, Jiri and Sebe, Nicu and Welling, Max",
title="DAPs: Deep Action Proposals for Action Understanding",
bookTitle="Computer Vision -- ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III",
year="2016",
publisher="Springer International Publishing",
address="Cham",
pages="768--784",
isbn="978-3-319-46487-9",
doi="10.1007/978-3-319-46487-9_47",
url="http://dx.doi.org/10.1007/978-3-319-46487-9_47"
}
```

If you like this project, give us a :star: in the github banner :wink:.

# What is this?

This repo complements our [repo](https://github.com/escorciav/daps) with the
training pipeline used for our ECCV-2016 work.

# I want to use it, Where can I start?

You can find a complete example of our project
[here](daps/test/functional_test.sh).
It covers data pre-processing, training and inference.

After training, you can plug-in a model trained here in our clean and
simplified inference [project](https://github.com/escorciav/daps).

# Installation and Usage

## Install

We use conda for deployment. You can create an evironment for this project
with the `environment-proto-linux-x64.yml`.

*Dependencies:* gcc, CUDA, conda.

> Are you a bash user? Take a look at our `install.sh` script.

## Usage

Once all the dependencies are installed, you are ready to go.

- Most of the scripts and tools should be executed from the project folder.

- You can add the project dir to your `PYTHONPATH` and use our modules as
standard python package.

> Are you bash & modules-env user? Take a look at our `activate.sh` script.

# What is not here?

This repo does not extract the C3D feature vector representation of your
videos. However, we try to alleviate your pain installing and assembling
all the cumulus of data such that you do not start from scratch.

- Take a look of
[this branch](https://github.com/escorciav/C3D/tree/setup-conda) for help to
install C3D.

- Do you want to extract C3D from frames? Take a look at
[this tool](tools/batch_frame_extraction.py) to simplify your life extracting
frames from videos.

- Extract C3D for frames without modifying prototxt manually. Take a look at
[this](tools/c3d_feat_frm.py)
