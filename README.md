[![GitHub issues](https://img.shields.io/github/issues/pldelisle/machine-learning-docker-environment.svg)](https://github.com/pldelisle/machine-learning-docker-environment/issues) [![GitHub stars](https://img.shields.io/github/stars/pldelisle/machine-learning-docker-environment.svg)](https://github.com/pldelisle/machine-learning-docker-environment/) [![GitHub forks](https://img.shields.io/github/forks/pldelisle/machine-learning-docker-environment.svg)](https://github.com/pldelisle//machine-learning-docker-environment/network) ![GitHub license](https://img.shields.io/badge/license-MIT-yellow.svg) [![Pulls on Docker Hub](https://img.shields.io/docker/pulls/pldelisle/machine-learning-docker-environment.svg)](https://hub.docker.com/r/pldelisle/machine-learning-environment/) [![Docker Stars](https://img.shields.io/docker/stars/pldelisle/machine-learning-environment.svg)](https://hub.docker.com/r/pldelisle/machine-learning-environment/)

# Machine Learning Docker Environment 
<img src="images/chip.png" width="96" height="96" vertical-align="bottom">

### Introduction

This is the Git repo of the Docker official image for a fully automated machine learning Docker environment. This environment has been primarily built for the [GTI770 Machine Learning class](https://en.etsmtl.ca/Programmes-Etudes/1er-cycle/Fiche-de-cours?Sigle=GTI770) at [Ecole de technologie superieure](https://en.etsmtl.ca/), but can be used for any other work related to machine learning and image processing. Two versions are available : 

* One with all libraries compiled and built for using NVIDIA GPUs
* One with all libraries compiled and built to only use a x86-64 CPU

### Supported tags and respective `Dockerfile` links 

`[latest-gpu]` *[Dockerfile](https://github.com/pldelisle/machine-learning-environment/blob/master/gpu/Dockerfile)*

`[latest-cpu]` *[Dockerfile](https://github.com/pldelisle/machine-learning-environment/blob/master/cpu/Dockerfile)*

### Quick references

* Maintained by: 

	[Pierre-Luc Delisle](https://github.com/pldelisle) 

* Where to file issues: 
	
	[Github issues](https://github.com/pldelisle/machine-learning-docker-environment/issues)

* Supported architectures:

	`[amd64]`[]() `[amd64-nvidia]`

* Supported Docker versions:
	Docker Community Edition 1.9 with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin.

* Base image:
	[NVIDIA's Ubuntu 16.04 CUDA 8.0 cuDNN 6 devel](https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/8.0/devel/cudnn6/Dockerfile)

* Docker Hub URL : [Docker Hub](https://hub.docker.com/r/pldelisle/machine-learning-environment/)


### Included software in Docker container 

#### Machine learning frameworks

* [TensorFlow 1.4](http://tensorflow.org) with [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
* [Caffe 1.0](http://caffe.berkeleyvision.org) 
* [Theano](https://github.com/Theano/Theano)
* [scikit-learn](http://scikit-learn.org/stable/)
* [pytorch](https://github.com/pytorch/pytorch)

#### Image processing  

* [OpenCV 3.3.1](https://github.com/opencv/opencv)
* [scikit-image](http://scikit-image.org)
* [Pillow](https://python-pillow.org)

#### Included Tensorflow wrappers 

* [tflearn](https://github.com/tflearn/tflearn)
* [keras](https://keras.io)

#### Other utility libraries

 * [graphviz](http://www.graphviz.org)
 * [matplotlib](http://matplotlib.org)
 * [scipy](https://www.scipy.org)
 * [jupyter](http://jupyter.org)
 * [sphinx](http://www.sphinx-doc.org/en/stable/)
 * [pytest](https://docs.pytest.org/en/latest/)

### Minimum requirements

### For dockerfile.gpu

* NVIDIA GeForce 700 or above GPU
* 10 GB free hard disk space

### For dockerfile.cpu

* 5 GB free hard disk space

### Notes

OpenCV, Tensorflow, Theano and Caffe are built with NVIDIA GPU support for hardware acceleration. 

OpenCV has been compiled for Python3. The `ml_venv` python virtual environment contains ready to use OpenCV Python3 binding library.

Port `6006` is exposed for Tensorboard log file parsing. 


### Usage

#### Getting started (GPU version)

`$ docker pull pldelisle/machine-learning-environment`

`$ nvidia-docker create --name gti770_env --volume /home/<your username>/path/to/shared/files pldelisle/machine-learning-environment:latest-gpu`

`$ nvidia-docker start gti770_env`

#### Getting started (CPU version)

`$ docker pull pldelisle/machine-learning-environment`

`$ docker create --name gti770_env --volume /home/<your username>/path/to/shared/files pldelisle/machine-learning-environment:latest-cpu`

`$ docker start gti770_env`

#### To get root access to the container 

`$ nvidia-docker exec -u root -it gti770_env /bin/bash`

#### To access to prebuilt virtualenv with installed libraries

`$ source /home/ubuntu/ml_venv/bin/activate`
`(ml_venv)$ `

### How to contribute ?
- [X] Create a branch by feature and/or bug fix
- [X] Get the code
- [X] Commit and push
- [X] Create a pull request

#### Branch naming

##### Feature branch
> feature/ [Short feature description] [Issue number]

##### Bug branch
> fix/ [Short fix description] [Issue number]

##### Commits syntax:

##### Adding code:
> \+ Added [Short Description] [Issue Number]

##### Deleting code:
> \- Deleted [Short Description] [Issue Number]

##### Modifying code:
> \* Changed [Short Description] [Issue Number]

##### To build the container : 
`docker build -t pldelisle/machine-learning-environment:latest-cpu --compress --rm .`  
`nvidia-docker build -t pldelisle/machine-learning-environment:latest-gpu --compress --rm .`

### Credits

<div>Icons made by <a href="http://www.freepik.com" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a> is licensed by <a href="http
://creativecommons.org/licenses/by/3.0/" title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a></div>
