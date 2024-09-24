# Learning Deep Equihash Functions
This repository is the official implementation of my thesis: “Learning Deep Equihash Functions.”

## Installation
```setup
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Datasets
The OpenImages mosaics are constructed from the OpenImage dataset. Each of the three splits (“train,” “valid,” and “test”) contains 100K images taken from OpenImages and cropped to 180x180 at their center. 

1) The [train split](https://drive.google.com/file/d/1_2nNEEvzuDNFxm8_eXe_dTXC72ZRd3g4/view?usp=sharing) needs to be at `data/datasets/OpenImages/hdf5/train.hdf5`. (9.5 GB)
2) The [valid split](https://drive.google.com/file/d/1_6BxIQvjGchpHuKv6nVRVCCa983Df8Qj/view?usp=sharing) needs to be at `data/datasets/OpenImages/hdf5/valid.hdf5`. (9.5 GB)
3) The [test split](https://drive.google.com/file/d/1Vw_IHYQpTPWETytw1BkbtR1dGYumCiBo/view?usp=sharing) needs to be at `data/datasets/OpenImages/hdf5/test.hdf5`. (9.5 GB)

After merging two images, three layers of noise are applied to the 2x1 mosaic. We use Perlin noise for one of these layers (the spatial distortion), which we precomputed for efficiency. The [pre-computed Perlin noise](https://drive.google.com/file/d/1Vnj9C7GiTeRQTj8Tvf4alHckOphr68ZI/view?usp=sharing) (6 GB) needs to be at `data/datasets/PerlinNoise/perlin180x270.hdf5`.

The NoisyMnist mosaics are constructed from the [MNIST dataset](https://drive.google.com/file/d/1_9hENa-Zh58osZW6sZTIIo-tjFkZ28N7/view?usp=sharing) (50 MB). This file contains the 70K MNIST images we used for the three splits and needs to be at `data/datasets/Mnsit/mnist.hdf5`.

## Training, Encoding, Indexing, and Evaluating
We provide all the scripts required for the project. Every script has a `--help` option describing its parameters.

1) Train the model with `train.py`
2) Encode a database into fingerprints (and the queries) w.r.t. a specific model with `encode.py`
3) Build the inverted index from a database of fingerprints with `build_index.py`
4) Evaluate the inverted index on those queries with `evaluate_binary_equihash.py`

> [!WARNING]
> We used an NVIDIA Quadro RTX 8000, which has 48 GB of RAM. The `train.py` and `encode.py` scripts were designed to work on this graphic card. These scripts might need to be modified to accommodate a smaller graphic card.

> [!TIP]
> When evaluating a model on an enormous database, splitting the work across multiple GPUs when encoding the database is very practical. The `encode.py` script provides the necessary options to encode a smaller portion of the database. The `merge.py` script merges the fingerprints of these smaller portions together.

To construct a new database, we need to generate the files used for the procedural generation of our database and the evaluation of our model on that database. This can be done with the `generate_labels.py` script.

## Replications of Our Results on OpenImages Mosaic
The first step in replicating our results is to train the models. This can be done with the following command:
```
python train.py OpenImagesMosaic ShannonHamming -e
python train.py OpenImagesMosaic HashNet -e
python train.py OpenImagesMosaic JMLH -e
```
It took us less than two weeks to train each model. The next step is to encode the database. We used a database containing 1 billion mosaics to evaluate our model in a large-scale scenario. These mosaics are procedurally generated to save memory. However, the indexes of the two images used in each mosaic must be generated in advance to avoid expensive computation when evaluating. To perform these evaluations, the pre-computed set of indexes (for the database, positive queries, and negative queries) must be generated with the following command:
```
python generate_labels.py OpenImagesMosaic 1BDB -b100000 -d1000000000 -p1000000 -n1000000 -t1000000 -m 1 2
```
This command will create the `data/labels/OpenImages` folder in the current directory and create the necessary files in a few minutes. Afterward, we need to encode the database with each model:
```
python encode.py OpenImagesMosaic 1BDB ShannonHamming --build_index
python encode.py OpenImagesMosaic 1BDB HashNet --build_index
python encode.py OpenImagesMosaic 1BDB JMLH --build_index
```
Each of these commands can take many days. If multiple GPUs are available, it is possible to split the work with the `--job_id` and `--nb_jobs` options. For example,
```
python encode.py OpenImagesMosaic 1BDB ShannonHamming --job_id 3 --nb_jobs 10
```
will partition the database into 10 splits and only encode the 4th one (job_id starts at 0). After encoding the 10 splits, we need to merge them with:
```
python merge.py OpenImagesMosaic 1BDB ShannonHamming --nb_jobs 10 --build_index
```
Note the usage of `--build_index`, which creates the inverted index (the hash table) used for retrieval. Finally, to evaluate the model on the encoded database, we executed:
```
python evaluate_binary_equihash.py OpenImagesMosaic 1BDB ShannonHamming
python evaluate_binary_equihash.py OpenImagesMosaic 1BDB HashNet
python evaluate_binary_equihash.py OpenImagesMosaic 1BDB JMLH
```

### Pre-Trained Models
To avoid training the models, we provide the links to the three OpenImages Mosaic models we trained: [ShannonHamming](https://drive.google.com/file/d/1_RALwPs5vqlkDinnQH_saM1wNk5o-QcT/view?usp=sharing) (131 MB), [HashNet](https://drive.google.com/file/d/1_M3W64BSJRt86A2qiGO2YJiaygRSfr0P/view?usp=sharing) (143 MB), [JMLH](https://drive.google.com/file/d/1_G9PYcrnnb23UPvBu3dSvSqdVl1yeqRV/view?usp=sharing) (136 MB).

They need to be in the following directories:
```
data/experiments/OpenImagesMosaic/ShannonHamming/main/current
data/experiments/OpenImagesMosaic/HashNet/main/current
data/experiments/OpenImagesMosaic/JMLH/main/current
```

## Replications of Our Results on NoisyMnist Mosaic
Replicating our results on NoisyMnist Mosaic is similar to replicating our results on OpenImages Mosaic. First, the following script must be executed to create the necessary file for the database of the 10 million mosaics we used (to evaluate the Irrelevant Collision Rate (ICR):
```
python generate_labels.py FastMnistMosaic 10MDB -b10000 -d10000000 -p1000000 -n1000000 -t1000000 -m 2 2
```
The main difference compared to OpenImagesMosaic is that there are many variants of each model. For example, the command
```
python train.py FastMnistMosaic ShannonHamming -v lambda_ablation -i5 -e --database_name 10MDB
```
will train on the 6th variant (i starts at 0) of the `lambda_ablation`. The variants are specified in `configs/FastMnistMosaic/ShannonHamming/lambda_ablation.json` where the 5th row of data specifies that this run will be executed with the seed 3210, a lambda of 0.006 and will be named s3210_l0.006. Each model has many variants. They can all be found in the appropriate folder:
```
configs/FastMnistMosaic/ShannonHamming/
configs/FastMnistMosaic/HashNet/
configs/FastMnistMosaic/JMLH/
```
> [!NOTE]
> The `--database_name 10MBD` option will encode, build the inverted index, and evaluate the model on the `10MDB` database after the training is completed. This is practical since this database and the neural network used are small.

## Expanding our Work
It is easy to modify the hyperparameters of each model. Suppose we want to train the JMLH model on NoisyMnist Mosaic using a Kullback–Leibler divergence of 0.2 (instead of the default 0.1). By modifing the value associated with "kld_coeff" inside `configs/FastMnistMosaic/JMLH/modelconfig.json`, we can set the Kullback–Leibler divergence to any value. However, this will affect every variant that does not explicitly set this value. To avoid this problem, we can create a new variant config, say `configs/FastMnistMosaic/JMLH/kld_ablation.json`, and overwrite the default kld_coeff with a header "trainer_kwargs:loss_kwargs:kld_coeff" and setting the value we want in the following row. Refer to `configs/FastMnistMosaic/JMLH/nb_classes_ablation.json` for an example where we modify the number of classes used for training.


For new models using a new loss or a new training loop, you will need to create a new class. For an example, refer to `equihash/trainers/hashnet_trainer.py`, where the training loop for HashNet is defined in the HashNetTrainer class. This class needs to maintain a training_log and implement `train`, `aggregate`, `state_dict`, and `load_state_dict`. Furthermore, when creating a new `modelconfig.json` for your new model, you will need to set "trainer_class" and "trainer_kwargs" appropriately. See `configs/FastMnistMosaic/HashNet/modelconfig.json` for an example.
