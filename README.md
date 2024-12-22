# Learning Deep Equihash Functions
This repository contains the official implementation of **Learning Deep Equihash Functions**, as detailed in my thesis.

## Table of Contents
  - [Installation](README.md#installation)
  - [Datasets](README.md#datasets)
  - [Usage](README.md#usage)
  - [Replicating Results](README.md#replicating-results)
      - [OpenImages Mosaic](README.md#openimages-mosaic)
      - [NoisyMnist Mosaic](README.md#noisymnist-mosaic)
  - [Pre-Trained Models](README.md#pre-trained-models)
  - [Extending This Work](README.md#extending-this-work)

## Installation
Install the required dependencies and set up the environment with:
```setup
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Datasets
### OpenImages Mosaics
The dataset is available on Hugging Face [duchema/open-images-100k-180x180](https://huggingface.co/datasets/duchema/open-images-100k-180x180). It is built from the [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html). Each split (train, valid, and test) contains 100K images, cropped to 180x180 pixels. Ensure the three HDF5 files are located at `data/datasets/OpenImages/hdf5/`.

To construct a 2x1 mosaic, we add three layers of noise, one of which uses [precomputed Perlin noise](https://drive.google.com/file/d/1Vnj9C7GiTeRQTj8Tvf4alHckOphr68ZI/view?usp=sharing) (6 GB). Place the file in `data/datasets/PerlinNoise/perlin180x270.hdf5`.

### NoisyMnist Mosaic
The NoisyMnist mosaic dataset is constructed from the [MNIST dataset](https://drive.google.com/file/d/1_9hENa-Zh58osZW6sZTIIo-tjFkZ28N7/view?usp=sharing) (50 MB). This file contains the 70K MNIST images in the HDF5 format. Ensure this file is located at `data/datasets/Mnist/mnist.hdf5`.

## Usage
This repository provides scripts for the primary tasks:
1) **Training**: Train a model with `train.py`
2) **Database**: Generate the labels of a database and queries with `generate_labels.py`
3) **Encoding**: Encode the database and the set of queries into fingerprints using the trained model with `encode.py`
4) **Indexing**: Build a hash table-based inverted index for efficient search with `build_index.py`
5) **Evaluation**: Evaluate the index on a database w.r.t. a set of queries with `evaluate_binary_equihash.py`

Each script supports a `--help` option for parameter details.
> [!NOTE]
> Although the database and queries are procedurally generated, each query’s ground truth must be precomputed and saved to evaluate the models efficiently. This is why we precompute and save the labels with `generate_labels.py`.

> [!WARNING]
> These scripts were designed to run on an NVIDIA Quadro RTX 8000 (48 GB of VRAM). Modifications may be needed to run the `train.py` and `encode.py` scripts on lower-capacity GPUs.

> [!TIP]
> For large datasets, distribute the encoding step across multiple GPUs. Use `encode.py` options `--job_id` and `--nb_jobs` to divide the workload and use `merge.py` to combine the results.

## Replicating Results
### OpenImages Mosaic
1) **Training**: To replicate our results, train the models as follows:
```bash
python train.py OpenImagesMosaic ShannonHamming -e
python train.py OpenImagesMosaic HashNet -e
python train.py OpenImagesMosaic JMLH -e
```
Training each model takes one to two weeks.

2) **Generating Labels**: Before evaluation, create a pre-computed set of labels for the database and queries:
```bash
python generate_labels.py OpenImagesMosaic 1BDB -b100000 -d1000000000 -p1000000 -n1000000 -t1000000 -m 1 2
```
This command generates the necessary labels in the `data/labels/OpenImages` directory. Here, `1BDB` specifies the database name, while `-d1000000000` indicates that the database consists of 1 billion mosaics. The `-p` and `-n` flags define the numbers of positive and negative queries, respectively, and `-t` specifies the number of triplets (each consisting of an anchor, positive, and negative) used for efficient evaluation of the relevant-collision rate and irrelevant-collision rate.

3) **Encoding**: Encode the database with each model:
```bash
python encode.py OpenImagesMosaic 1BDB ShannonHamming --build_index
python encode.py OpenImagesMosaic 1BDB HashNet --build_index
python encode.py OpenImagesMosaic 1BDB JMLH --build_index
```

Encoding can take several days. To parallelize, use `--job_id` and `--nb_jobs` to partition the work. For example:
```
python encode.py OpenImagesMosaic 1BDB ShannonHamming --job_id 3 --nb_jobs 10
```
will partition the database into 10 splits and only encode the 4th one (job_id starts at 0). After encoding the 10 splits, we need to merge them with:
```
python merge.py OpenImagesMosaic 1BDB ShannonHamming --nb_jobs 10 --build_index
```
> [!NOTE]
> The `-build_index` flag automatically triggers `build_index.py` after computing the fingerprints, streamlining the workflow by building the index in the same step.

4) Evaluation: Evaluate each model on the encoded database:
```
python evaluate_binary_equihash.py OpenImagesMosaic 1BDB ShannonHamming
python evaluate_binary_equihash.py OpenImagesMosaic 1BDB HashNet
python evaluate_binary_equihash.py OpenImagesMosaic 1BDB JMLH
```

## NoisyMnist Mosaic
Replication on NoisyMnist follows a similar process to OpenImages Mosaic but includes additional model variants. Generate the 10 million labels with:
```
python generate_labels.py FastMnistMosaic 10MDB -b10000 -d10000000 -p1000000 -n1000000 -t1000000 -m 2 2
```
To train a model variant:
```
python train.py FastMnistMosaic ShannonHamming -v lambda_ablation -i5 -e --database_name 10MDB
```
will train on the 6th variant (i starts at 0) for the `lambda_ablation`. The variants are specified in `configs/FastMnistMosaic/ShannonHamming/lambda_ablation.json` where the 5th row of data specifies that this run will be executed with the seed 3210, a lambda of 0.006 and will be named s3210_l0.006. Each model has many variants. They can all be found in the appropriate folder:
```
configs/FastMnistMosaic/ShannonHamming/
configs/FastMnistMosaic/HashNet/
configs/FastMnistMosaic/JMLH/
```
> [!NOTE]
> The `--database_name 10MBD` option will encode, build the inverted index, and evaluate the model on the `10MDB` database after the training is completed.

### Pre-Trained Models
We provide three pre-trained models together with their optimizer and training evaluations for OpenImages Mosaic:
  - [ShannonHamming](https://drive.google.com/file/d/1_RALwPs5vqlkDinnQH_saM1wNk5o-QcT/view?usp=sharing) (131 MB)
  - [HashNet](https://drive.google.com/file/d/1_M3W64BSJRt86A2qiGO2YJiaygRSfr0P/view?usp=sharing) (143 MB)
  - [JMLH](https://drive.google.com/file/d/1_G9PYcrnnb23UPvBu3dSvSqdVl1yeqRV/view?usp=sharing) (136 MB)

Place each in the corresponding directory:
```
data/experiments/OpenImagesMosaic/ShannonHamming/main/current
data/experiments/OpenImagesMosaic/HashNet/main/current
data/experiments/OpenImagesMosaic/JMLH/main/current
```

## Extending This Work
### Modifying Hyperparameters
Suppose we want to train the JMLH model on NoisyMnist Mosaic using a Kullback–Leibler divergence of 0.2 (instead of the default 0.1). By modifing the value associated with "kld_coeff" inside `configs/FastMnistMosaic/JMLH/modelconfig.json`, we can set the Kullback–Leibler divergence to any value. However, this will affect every variant that does not explicitly set this value. To avoid this problem, we can create a new variant config, say `configs/FastMnistMosaic/JMLH/kld_ablation.json`, and overwrite the default kld_coeff with a header "trainer_kwargs:loss_kwargs:kld_coeff" and setting the value we want in the following row. Refer to `configs/FastMnistMosaic/JMLH/nb_classes_ablation.json` for an example where we modify the number of classes used for training.

### Adding New Models
For new models using a new loss or a new training loop, you will need to create a new class. For an example, refer to `equihash/trainers/hashnet_trainer.py`, where the training loop for HashNet is defined in the HashNetTrainer class. This class needs to maintain a `training_log` and implement `train`, `aggregate`, `state_dict`, and `load_state_dict`. Furthermore, when creating a new `modelconfig.json` for your new model, you will need to set "trainer_class" and "trainer_kwargs" appropriately. See `configs/FastMnistMosaic/HashNet/modelconfig.json` for an example.
