# Learning an Equihash to Find a Needle in a Haystack
This repository is the official implementation of Learning an Equihash to Find a Needle in a Haystack

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
python setup.py build_ext --inplace
```

Download ProcDB files (~25GB) to train a model from scratch or encode the database and queries:
```sh
pip install gdown
gdown --folder 1voIlQpQYcY-V8KTZOCOTHsdpbCNHtaRy
```

## Equihash training and evaluation pipeline
1) Train the model with ```train.py```
2) Encode the database with ```encode_mosaics.py```
3) Build the inverted index with ```build_index.py```
4) Encode the queries, again with ```encode_mosaics.py```
5) Evaluate the inverted index on those queries with ```eval_index.py```

## Training
To train the models in the paper, run those commands:
```sh
python train.py configs/natural_mosaic_shannon_hamming_64c3.json -p `seq 0 5000 100000` -f 100 -e -D 51966
python train.py configs/natural_mosaic_shannon_hamming_64c3_32c4.json -p `seq 0 5000 100000` -f 100 -e -D 51966
python train.py configs/natural_mosaic_hashnet_0.1a.json -p `seq 0 5000 100000` -f 100 -e -D 51966
python train.py configs/natural_mosaic_hashnet_0.15a.json -p `seq 0 5000 100000` -f 100 -e -D 51966
python train.py configs/natural_mosaic_hashnet_0.2a.json -p `seq 0 5000 100000` -f 100 -e -D 51966
```
With the training seed set to 51966 (0xCAFE), the networks should be exactly the same as in the paper.
Those commands will create checkpoints in the `states/` folder. Each network takes about one week to train and requires a GPU with at least 30GB of memory.

Use `python train.py --help` for more options. The configs used in the paper are in `configs/`. To train a different model, you will need to create a new config file.

## Encoding the ProcDB database
To use the same database as in the paper, use the precedural seed 2766 (0xACE). We set ```-w valid``` to use the validation image bank.
```sh
python encode_mosaics.py configs/natural_mosaic_shannon_hamming_64c3.json -l100000 -iimages/natural_index1Bx2.hdf5 -wvalid -D2766
python encode_mosaics.py configs/natural_mosaic_shannon_hamming_64c3_32c4.json -l100000 -iimages/natural_index1Bx2.hdf5 -wvalid -D2766
python encode_mosaics.py configs/natural_mosaic_hashnet_0.2a.json -l100000 -iimages/natural_index1Bx2.hdf5 -wvalid -D2766
python encode_mosaics.py configs/natural_mosaic_hashnet_0.15a.json -l100000 -iimages/natural_index1Bx2.hdf5 -wvalid -D2766
python encode_mosaics.py configs/natural_mosaic_hashnet_0.1a.json -l100000 -iimages/natural_index1Bx2.hdf5 -wvalid -D2766
```
Each command takes about 160 hours (with a GPU) and produce a 8GB hdf5 file in `fingerprints/`. Those can be parallelized with the `-n` and `-j` option, use `python encode_mosaics.py --help` for more informations.
Afterward, if `-n` and `-j` are used, you need to merge the chunks together (run `python merge.py --help`).

## Building the inverted index
To build the database's inverted index, run:
```sh
python build_index.py configs/natural_mosaic_shannon_hamming_64c3.json -l100000 -wvalid -D2766
python build_index.py configs/natural_mosaic_shannon_hamming_64c3_32c4.json -l100000 -wvalid -D2766
python build_index.py configs/natural_mosaic_hashnet_0.2a.json -l100000 -wvalid -D2766
python build_index.py configs/natural_mosaic_hashnet_0.15a.json -l100000 -wvalid -D2766
python build_index.py configs/natural_mosaic_hashnet_0.1a.json -l100000 -wvalid -D2766
```
Each commands takes about 5 minutes and create a 20GB hdf5 file in `indexes/`.
Use `python build_index.py --help` for more options.

## Encoding the queries
Similar to encoding the database, however we set `-L1000000` to only encode the first 1M queries. Furthermore, we use the seed 3054 (0xBEE). If we use 2766, the queries will be exactly the same as the database (and every model would have 100% recall). 
```sh
python encode_mosaics.py configs/natural_mosaic_shannon_hamming_64c3.json -l100000 -iimages/natural_index1Bx2.hdf5 -wvalid -D3054 -L1000000
python encode_mosaics.py configs/natural_mosaic_shannon_hamming_64c3_32c4.json -l100000 -iimages/natural_index1Bx2.hdf5 -wvalid -D3054 -L1000000
python encode_mosaics.py configs/natural_mosaic_hashnet_0.2a.json -l100000 -iimages/natural_index1Bx2.hdf5 -wvalid -D3054 -L1000000
python encode_mosaics.py configs/natural_mosaic_hashnet_0.15a.json -l100000 -iimages/natural_index1Bx2.hdf5 -wvalid -D3054 -L1000000
python encode_mosaics.py configs/natural_mosaic_hashnet_0.1a.json -l100000 -iimages/natural_index1Bx2.hdf5 -wvalid -D3054 -L1000000
```
Each command takes about 10 minutes (with a GPU) and create a 8MB hdf5 file in `fingerprints/`.

## Evaluation of the equihash
To evaluate the models, run:
```sh
python eval_index.py -Iindexes/natural_mosaic_shannon_hamming_64c3_step_100000_valid_index_2766.hdf5 -Qfingerprints/natural_mosaic_shannon_hamming_64c3_step_100000_valid_1000000fingerprints_3054.hdf5 -Sresults/natural_mosaic_shannon_hamming_64c3_step_100000_valid.pkl
python eval_index.py -Iindexes/natural_mosaic_shannon_hamming_64c3_32c4_step_100000_valid_index_2766.hdf5 -Qfingerprints/natural_mosaic_shannon_hamming_64c3_32c4_step_100000_valid_1000000fingerprints_3054.hdf5 -Sresults/natural_mosaic_shannon_hamming_64c3_32c4_step_100000_valid.pkl
python eval_index.py -Iindexes/natural_mosaic_hashnet_0.2a_step_100000_valid_index_2766.hdf5 -Qfingerprints/natural_mosaic_hashnet_0.2a_step_100000_valid_1000000fingerprints_3054.hdf5 -Sresults/natural_mosaic_hashnet_0.2a_step_100000_valid.pkl
python eval_index.py -Iindexes/natural_mosaic_hashnet_0.15a_step_100000_valid_index_2766.hdf5 -Qfingerprints/natural_mosaic_hashnet_0.15a_step_100000_valid_1000000fingerprints_3054.hdf5 -Sresults/natural_mosaic_hashnet_0.15a_step_100000_valid.pkl
python eval_index.py -Iindexes/natural_mosaic_hashnet_0.1a_step_100000_valid_index_2766.hdf5 -Qfingerprints/natural_mosaic_hashnet_0.1a_step_100000_valid_1000000fingerprints_3054.hdf5 -Sresults/natural_mosaic_hashnet_0.1a_step_100000_valid.pkl
```
Use `python eval_index.py --help` for more options. Each command will produce one row of the paper's Table 1.

The `-S` options will save the following results w.r.t. each queries:
  1) Was it a perfect retrieval
  2) Was the relevant data in the bucket
  3) The bucket size
Those are used to create the graphs of the paper. (see notebooks/graphs.ipynb)


## Pre-trained models
To download the trained models, run:
```sh
cd states
gdown 1tsDqsF8f5Gudxnp7yi2u7zqSs11JL336 #for natural_mosaic_shannon_hamming_64c3_step_100000.pth
gdown 1r8YrdOsznJNN66pSMU2uqdtQXCKiR8eM #for natural_mosaic_shannon_hamming_64c3_32c4_step_100000.pth
gdown 1QIIAxzNu32zbW2KduDUq5CRUIgW-P9P5 #for natural_mosaic_hashnet_0.1a_step_100000.pth
gdown 1tJDReM7bF5wQhi70TR2p1v6ryXjEXoRM #for natural_mosaic_hashnet_0.15a_step_100000.pth
gdown 1P7JmWbYxfJHiW-M7RP_6oc3DVzxAmO-t #for natural_mosaic_hashnet_0.2a_step_100000.pth
```

## Pre-encoded database
To download the encoded fingerprints of the database, run:
```sh
cd fingerprints
gdown 1tcGvEltvP25DySyilJnnQs2qmWTUfwFZ #for natural_mosaic_shannon_hamming_64c3_step_100000_valid_fingerprints_2766.hdf5
gdown 1p_dAbxfzZg6pgu9lWwzUQAnlKk22mE0l #for natural_mosaic_shannon_hamming_64c3_32c4_step_100000_valid_fingerprints_2766.hdf5
gdown 1uirse-3n8u0znGjdHiH3LlIYbMq4DMaJ #for natural_mosaic_hashnet_0.1a_step_100000_valid_fingerprints_2766.hdf5
gdown 16cL0F8OwYAV8S5oIwoivRiQ6RfLkQhrw #for natural_mosaic_hashnet_0.15a_step_100000_valid_fingerprints_2766.hdf5
gdown 17faNLAmJERjxQziQZnb8jAlZPC9jQblQ #for natural_mosaic_hashnet_0.2a_step_100000_valid_fingerprints_2766.hdf5
```

## Pre-encoded queries
To download the encoded fingerprints of the queries, run:
```sh
cd fingerprints
gdown 1ZQgIg1bKSF_5co2QmG2sSH1ykO0J90S3 #for natural_mosaic_shannon_hamming_64c3_step_100000_valid_1000000fingerprints_3054.hdf5
gdown 1nG3hM91U1rxdlN2Gtgpd4BEsDB7Nacox #for natural_mosaic_shannon_hamming_64c3_32c4_step_100000_valid_1000000fingerprints_3054.hdf5
gdown 1X3-H6kXKmGeCLcx5iaw7wb_wkNpl6DPJ #for natural_mosaic_hashnet_0.1a_step_100000_valid_1000000fingerprints_3054.hdf5
gdown 1gJxzztNWR3SuSNLDeiNh2Ib9_q0MkZz5 #for natural_mosaic_hashnet_0.15a_step_100000_valid_1000000fingerprints_3054.hdf5
gdown 1uAOAVvpwLL0y_sGpGHS3mP9OqAWOaQqI #for natural_mosaic_hashnet_0.2a_step_100000_valid_1000000fingerprints_3054.hdf5
```

## Results

Table 1: Perfect retrieval rates on ProcDB. Error bars are below 0.1%
| Database size      | 1M              | 10M            | 100M           | 1B             |
| ------------------ |---------------- | -------------- | -------------- | -------------- |
| SH (64c3)          |     48.0%       |      47.2%     |      43.3%     |      31.5%     |
| SH (64c3+32c4)     |     46.7%       |      46.6%     |      45.8%     |      42.7%     |
| HashNet (α=0.2)    |     26.7%       |      19.3%     |      11.3%     |       5.1%     |
| HashNet (α=0.15)   |     28.7%       |      17.9%     |       8.3%     |       2.9%     |
| HashNet (α=0.1)    |     11.8%       |       3.7%     |       1.0%     |       0.2%     |
