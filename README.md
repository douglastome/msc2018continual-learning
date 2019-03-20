# A Broad and Deep Analysis of Elastic Weight Consolidation in Large Convolutional Neural Networks

This repository contains the code used to perform the experiments outlined in the final project of my MSc Computing Science.

In this work, we proposed a methodology to evaluate continual learning approaches consisting of a comprehensive and challenging experimental analysis involving large-capacity deep neural networks and the following multi-label classification tasks:
1) permutation learning
2) incremental class learning
3) unimodal sequential task learning
4) multi-modal sequential task learning

We illustrated this methodology using [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset) and [Elastic Weight Consolidation (EWC)](https://www.pnas.org/content/114/13/3521) as a case study.

## Running experiments

In order to clone the Anaconda environment necessary to run the code, create an environment using the provided `environment.yml` file:

`conda env create -f environment.yml`

Further instructions can be found in the Conda documentation [here](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

In addition, you will need to download the pre-trained VGGish checkpoint file: [vggish_model.ckpt](https://github.com/tensorflow/models/tree/master/research/audioset).

Next, you will need the original datasets used in the experiements. These are made publicly available by their respective authors:
1) [AudioSet](https://research.google.com/audioset/download.html)
2) [URBAN-SED](http://urbansed.weebly.com/)
3) [ESC-50](https://github.com/karoldvl/ESC-50)
4) [SVHN](http://ufldl.stanford.edu/housenumbers/)
5) [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
6) [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
7) [MNIST](http://yann.lecun.com/exdb/mnist/)

Note that you do not need to manually download MNIST since it can be downloaded via TensorFlow directly.

To train the provided network on one or a combination of the above datasets, first build the required dataset(s) by running `build_datasets.py`. After building each dataset, a separate .pkl will be saved in the specified directory.

If your experiment involves permutations of a dataset, run `build_dataset_permutations.py` to build the desired number of permutations of this dataset. Each permutation will be saved as a separate .pkl file.

Once you have built the desired dataset(s) and/or permuation(s) of a dataset, run `main.py` to train the network. Theare are a variety of command-line arguments that can be used to control the exact settings of your experiment. Please refer to `main.py` for a detailed description of each these arguments.
