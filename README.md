# Learning Deep Equihash Functions
This repository is the official implementation of my thesis: “Learning Deep Equihash Functions.”

## Installation
```setup
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Required Downloads to construct the OpenImages Mosaics
The OpenImages mosaics are constructed from the OpenImage dataset. Each of the three splits (“train,” “valid,” and “test”) contains 100K images taken from OpenImages and cropped to 180x180 at their center. 

1) The train split needs to be at `data/datasets/OpenImages/hdf5/train.hdf5`. (9.5 GB)
2) The valid split needs to be at `data/datasets/OpenImages/hdf5/valid.hdf5`. (9.5 GB)
3) The test split needs to be at `data/datasets/OpenImages/hdf5/test.hdf5`. (9.5 GB)

After merging two images, three layers of noise are applied to the 2x1 mosaic. We use Perlin noise for one of these layers (the spatial distortion), which we precomputed for efficiency.

1) This file needs to be at `data/datasets/PerlinNoise/perlin180x270.hdf5`.

Finally, we used a database containing 1 billion mosaics to evaluate our model in a large-scale scenario. These mosaics are procedurally generated to save memory. However, some files still need to be downloaded to evaluate our models on this database.

1) `this` keeps track of the two images we need to construct each mosaic in the database.
2) 12321

Each file must be in this folder: `data/datasets/labels/OpenImages`.

## Required Downloads to construct the NoisyMnist Mosaics
The NoisyMnist mosaics are constructed from the MNIST dataset. We use this HDF5 file. It contains the 70K MNIST images we used for the three splits and needs to be at `data/datasets/Mnsit/mnist.hdf5`.

Like OpenImages mosaics, the following files must be at `data/datasets/labels/OpenImages`.

## Training, Encoding, Indexing, and Evaluating
We provide all the scripts required for the project. Every script has a `--help` option describing its parameters.

1) Train the model with `train.py`
2) Encode a database into fingerprints (and the queries) w.r.t. a specific model with `encode.py`
3) Build the inverted index from a database of fingerprints with `build_index.py`
4) Evaluate the inverted index on those queries with `evaluate_binary_equihash.py`

Since the database is enormous, splitting it to work across multiple GPUs when encoding a database is very practical. The `encode.py` script provides the necessary option to do so. Afterward, the `merge.py` script can merge the outputs back together.

To construct a new database, we need to generate the files used for the procedural generation of our database and the evaluation of our model on that database. This can be done with the `generate_labels.py` script.

## Replications of Our Results
...

## Pre-Trained Models
The links to the pre-trained models

## Pre-Computed Inverted Index

## Pre-Encoded Queries
