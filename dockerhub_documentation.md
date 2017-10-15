## Supported tags and respective `Dockerfile` links 

`[latest-gpu]` *[Dockerfile](https://github.com/pldelisle/machine-learning-docker-environment/blob/master/dockerfile.gpu)*
`[latest-cpu]` *[Dockerfile](https://github.com/pldelisle/machine-learning-docker-environment/blob/master/dockerfile.cpu)*

## Quick references

* Maintained by: 

	[Pierre-Luc Delisle](https://github.com/pldelisle) 

* Where to file issues: 
	
	[Github issues](https://github.com/pldelisle/machine-learning-docker-environment/issues)

* Supported architectures:

	`[amd64]`[]() `[amd64-nvidia]`[](https://github.com/pldelisle/machine-learning-docker-environment/blob/master/dockerfile.gpu])

* Supported Docker versions:
	Docker Community Edition 1.9 with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin.

* Base image:
	[NVIDIA's Ubuntu 16.04 CUDA 8.0 cuDNN 6 devel](https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/8.0/devel/cudnn6/Dockerfile)


## Included softwares in Docker container 

# Machine learning frameworks

* [TensorFlow 1.3](http://tensorflow.org) with [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
* [Caffe 1.0](http://caffe.berkeleyvision.org) 
* [Theano](https://github.com/Theano/Theano)
* [scikit-learn](http://scikit-learn.org/stable/)
* [pytorch](https://github.com/pytorch/pytorch)

# Image processing  

* [OpenCV 3.3.0](https://github.com/opencv/opencv)
* [scikit-image](http://scikit-image.org)
* [Pillow](https://python-pillow.org)

# Included Tensorflow wrappers 

* [tflearn](https://github.com/tflearn/tflearn)
* [keras](https://keras.io)

# Other utility libraries

 * [graphviz](http://www.graphviz.org)
 * [matplotlib](http://matplotlib.org)
 * [scipy](https://www.scipy.org)
 * [jupyter](http://jupyter.org)
 * [sphinx](http://www.sphinx-doc.org/en/stable/)
 * [pytest](https://docs.pytest.org/en/latest/)

## Minimum requirements

# For dockerfile.gpu

* NVIDIA GeForce 700 or above GPU
* 10 GB free hard disk space

# For dockerfile.cpu

* X GB free hard disk space

# Notes

OpenCV, Tensorflow, Theano and Caffe are built with NVIDIA GPU support for hardware acceleration. 

OpenCV has been compiled for Python3. The `ml_venv` python virtual environment contains a ready to use OpenCV Python3 binding library.

Port `6006` is exposed for Tensorboard log file parsing. 


## Usage

`docker pull pldelisle/ml_env`

`nvidia-docker create pldelisle/ml_env --name < container name > --volume /home/< your username > /path/to/shared/files`

`nvidia-docker start gti770_ml`

# To get root access to the container 

`nvidia-docker exec -u root -it gti770_ml /bin/bash`

# To access to pre-built virtualenv with installed libraries

`source /home/ubuntu/ml_venv/bin/activate`
`(ml_venv)$`