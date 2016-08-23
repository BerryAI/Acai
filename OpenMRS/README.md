# Music Recommendation by BerryAI
This project is a music recommendation system developed by BerryAI.
## Getting Started
These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes
on how to deploy the project on a live system.
### Prerequisites Software
What things you need to install the software and how to install them
```
Python
Python-Librosa
Python-Lutorpy
Python-NumPy
Python-Scipy
Torch
Torch-cunn
Torch-dp
Torch-nn
Torch-optim
Torch-xlua
```
#### Linux
All major distributions of Linux provide packages for both Python and NumPy.
#### Mac OS X
```
pip install librosa
pip install lutorpy
pip install numpy
pip install scipy
```
#### Windows
Personally, I will recommend Anaconda as default Python compiler. To install
them, go to page
```
https://www.continuum.io/downloads
```
and find the proper install packages

#### Torch
You can find torch installation instruction in the official site: 
http://torch.ch/
You can also refer to our wiki page:
https://github.com/BerryAI/music_cortex/wiki/Torch-Setup

### Prerequisites Dataset
In this project, we use some public open database, and they are

* Last.fm 1k user data [download](http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz)
* Million Song Database[download](http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/unique_tracks.txt)
* Million Song Database subset [download](https://drive.google.com/file/d/0B7s9m90eW6dtMnk5Q2M1aFBfeDA/view?usp=sharing)
* Echo Nest user data [download](http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip)

For convenience purpose, I have calculate the intersection between 1k user data
and MSD Database. [HERE](https://drive.google.com/open?id=0B7s9m90eW6dtX084eTNXQ2NLblU)
is the download link.

### Installation


## Usage
Test functions are under ./test folder. After downloading all the data files,
please put the extracted files into ./data folder.

Then run collaborative filtering
```
python test_cf_hf_gd.py
```
run convolutional neural networks
```
th example.lua
```
in command line under the directory of the project installed.

## Algorithms included:

### Collaborative Filtering Methods

#### * Memory based recommendation

The recommendation equation is: <br />
![](https://upload.wikimedia.org/math/c/1/d/c1da0ee720e382372582a51ac2368925.png)

Where U is the full set of all users, ![](http://mathurl.com/hm6fwsr.png) is
 user u's rating score of item
i, and ![](http://mathurl.com/h7lc86c.png) is the average rating score for user
u. For similarity function ![](http://mathurl.com/gvgors5.png), we have two
approaching ways:

1. K nearest neighbours: <br />
![](http://mathurl.com/zgm3zlh.png)
Where ![](http://mathurl.com/jua8fgh.png) is the set of neighbors of user a.
2. Pearson Correlation: <br />
![](https://camo.githubusercontent.com/f1176f6282d9043a2104d01c208f9946e150db75/687474703a2f2f6d61746875726c2e636f6d2f686d37747865612e706e67)

#### * Matrix Factorization and Hidden Features
We could use much smaller dimension matrix P, Q to represent and approximate the
full rating score matrix R. That is: <br />
![](http://mathurl.com/jy3us2x.png)

Normally, we have two different approaches:

1. Singular Value Decomposition

R is a m*n matrix
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/3315de0d8549ccefd4c619e4e6cce6ba041dde3c)

Where:
* M is m*m unitary matrix
* Σ is m*n diagonal matrix with singular values
* N is n*n unitary matrix

With first k singular values, we could approximate R as: <br />
![](http://mathurl.com/znt89p3.png)

Then: <br />
![](http://mathurl.com/hn5gzlf.png)

2. Gradient Descent
We try to minimize the norm of residue matrix: <br />
![](http://latex.codecogs.com/gif.latex?%5Cmin_%7BP%2CQ%7D%20F%28P%2CQ%29%20%3D%20%5C%7CR%20-%20PQ%5ET%5C%7C_2)

we have two different approaches:
* Classic Gradient Descent: <br />
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/0154a26cc6ac60465f8eb3d00d2f2dfa6899da2a)

* Stochastic Gradient Descent with momentum: <br />
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/350886f1e3aaa6e9352caca8581274df95ac54e6)

Both methods will converge, but please be careful choosing coefficients.

### Convolutional Neural Networks

#### * Building Blocks

1.Convolutional Layer
Convolutional layers are the core building block of CNNs. The layer's parameters consist of sets of learnable filters/kernels ![](http://mathurl.com/hvjwhf8). During training, the parameters are learned from data in order to solve the target problem. The forward equation is: <br />
![](http://mathurl.com/jrpb8hx)
where ![](http://mathurl.com/zuaxm9z) is the data in layer L in filter i, and * represents convolution operation.

2.Max Pooling Layer
Pooling layer is another important concept of CNN. It is down-sampling process. Max pooling is a non-linear down-sampling method. The forward equation is: <br />
![](http://mathurl.com/hz25w8p)
where p,q are the pooling size.

3.Rectified Linear Units Layer
ReLU layers apply nonlinear activation function to neurons. Comparing to other common activation functions, ReLU is fast in training and suffers less on gradient extenuation during training.The forward equation is: <br />
![](http://mathurl.com/guq6vge)

4.Loss Layer
The loss layer is the last layer in CNN which defines the training deviation between real predicted results and target results.
We provide 2 options in our model.
* Mean Squared Error
![](http://mathurl.com/jcmhy4w)
where ![](http://mathurl.com/znt7jgr) is real output and is target output.

* Softmax Loss
![](http://mathurl.com/hul2kyg)
where ![](http://mathurl.com/znt7jgr) is real output and is target output.

#### * Back-propagation Rule
* Stochastic Gradient Descent with momentum: <br />
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/350886f1e3aaa6e9352caca8581274df95ac54e6)

## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## License
The OpenMRS source code and binaries are released under the [MIT license](../LICENSE.md)
