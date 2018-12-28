This repo contains the demo code for an unsupervised deep RE-ID model DECAMEL
(DEep Clustering-based Asymmetric MEtric Learning).
Our implementation was done using [matconvnet](https://github.com/vlfeat/matconvnet).
We provide an instruction for using our code, but do not encourage you to modify the source
because the structure of code dependence could be quite complex.
We recommand to use the [Market-1501 dataset](www.liangzheng.org/Project/project_reid.html) dataset for our demo.

Hong-Xing Yu, Ancong Wu, Wei-Shi Zheng, 
"Unsupervised Person Re-identification by Deep Asymmetric Metric Embedding",
IEEE Transactions on Pattern Analysis and Machine Intelligence. (DOI 10.1109/TPAMI.2018.2886878)

## Results on large popular datasets

Dataset| Rank-1| Rank-5| Rank-10| MAP
-|-|-|-|-
Market-1501| 60.24| 75.95| 81.12| 32.44
MSMT17| 30.34| 43.06| 49.10| 11.13

Note that the results could be a little bit different in your run due to random factors.

## Usage

#### Reproducing the reported results
1. Download the [Market-1501 dataset](www.liangzheng.org/Project/project_reid.html)
and put the unzipped folders into /data/market.
In other words, you will have a folder structure like:
- data
    - market
        - bounding_box_test
        - bounding_box_train
        - query
And then run /src/makeImdb.m

2. Enter /src, and run
```
main('gpus', YOUR_GPU)
```
where YOUR_GPU is an array of int, e.g. [1], [1,2].
Note that the index starts from 1 instead of 0.
If you use a single GPU, the default batch size (216) requires 11GB memory.

#### For different datasets
1. Prepare your data in the format like /data/market.mat:
- market.mat
    - images (1x1 struct)
        - data (HxWxCxN single, where H=144, W=56, C=3, N=dataset size)
        - labels (1xN single)
        - set (1xN uint8, in which 1=train, 3=gallery/test, 4=probe/query)
        - idxViews (1xN single. Index starts from 1.)

Please check our template /src/makeImdb_toy_market.m which forms a toy dataset toy_market.mat,
and you might want to modify it to suit your need.

2. Enter /src, and run
```
main('gpus', YOUR_GPU, 'data_path', DATAPATH)
```
where YOUR_GPU is an array of int, e.g. [1], [1,2],
DATAPATH is the file name of the dataset .mat file in the last step.
Note that the index starts from 1 instead of 0.
If you use a single GPU, the default batch size (216) requires 11GB memory.

## Reference
If you find our work helpful in your research, please kindly cite our paper:
```
@article{yu2018unsupervised,
title={Unsupervised Person Re-identification by Deep Asymmetric Metric Embedding},
author={Yu, Hong-Xing and Wu, Ancong and Zheng, Wei-Shi},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (DOI 10.1109/TPAMI.2018.2886878)},
}
```

If you have any questiones please feel free to contact me at xKoven@gmail.com