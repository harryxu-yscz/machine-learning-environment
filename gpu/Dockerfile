FROM nvidia/cuda:8.0-devel-ubuntu16.04
ARG PYTHON_VERSION=3.5
ARG NB_USER=ubuntu 
ENV OPENCV_DIR /opt/opencv 
ENV CAFFE_ROOT /opt/caffe
ENV LIBGPUARRAY_DIR /opt/libgpuarray
ENV NUM_CORES 8 
ENV NB_UID 1000 
ENV CUDNN_VERSION 6.0.21
ARG CLONE_TAG=1.0
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
ENV OPENCV_VERSION=3.3.1
ENV CUDA_ENABLED=ON
ENV OPENCL_ENABLE=OFF

LABEL maintainer "Pierre-Luc Delisle <pierre-luc.delisle.1@ens.etsmtl.ca>"
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
LABEL com.nvidia.caffe.version="0.15"

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER

RUN apt-get update && apt-get install -y --no-install-recommends \
	    libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
	    libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 \
        apt-utils \ 
        build-essential \
        ca-certificates \
        cmake \
        curl \
        doxygen \
        freeglut3* \
        g++ \
        gfortran \
        git \
        graphviz \
        libatlas-base-dev \
        libatlas-base-dev \ 
        libavcodec-dev \
        libavformat-dev \
        libboost-* \
        libcupti-dev \
        libcurl3-dev \
        libdc1394-22-dev \ 
        libeigen3-dev \  
        libfaac-dev \
        libfreetype6-dev \
        libgflags-dev \
        libgl1-mesa-dev \
        libgoogle-glog-dev \
        libgphoto2* \
        libgstreamer* \
        libgtk-3-dev \
        libgtk2.0-dev \ 
        libgtkglext1* \
        libhdf5-dev \
        libhdf5-serial-dev \
        libjasper-dev \
        libjpeg-dev \
        liblapacke-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopenblas-dev \ 
        libopencore-amrnb-dev \ 
        libopencore-amrwb-dev \ 
        libopencv-dev \ 
        libopenni2* \
        libpng-dev \
        libprotobuf-dev \
        libsnappy-dev \
        libswscale-dev \
        libtbb-dev \
        libtbb2 \
        libtheora-dev \
        libtiff-dev \
        libv4l-dev \
        libvorbis-dev \
        libzmq3-dev \
        libxvidcore-dev \
        mesa-common-dev \
        nano \
        ocl-icd-opencl-dev \ 
        openblas-* \
        openblas* \
        pkg-config \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy \
        python-setuptools \
        python3-dev \
        python3-h5py \
        python3-matplotlib \
        python3-nose \
        python3-numpy \
        python3-pandas \
        python3-scipy \
        python3-skimage \
        python3-sklearn \
        python3-sympy \
        python3-tk \
        software-properties-common \
        swig \
        unzip \
        v4l-utils \
        vim \
        wget \
        x264 \ 
        zip \
        zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
	curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - && \
	apt-get update && apt-get install tensorflow-model-server

RUN wget https://bootstrap.pypa.io/get-pip.py && \ 
    python3 get-pip.py && \ 
    pip3 install numpy \
                 virtualenv \
                 virtualenvwrapper \
                 boost \
                 pytest && \
    pip install numpy \ 
                virtualenv \
                virtualenvwrapper \
                boost \
                pytest

RUN mkdir -p /src && \
    mkdir -p $CAFFE_ROOT && \
    cd $CAFFE_ROOT && \
    git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    pip install --upgrade pip && \
    cd python && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    cd .. && \
    mkdir build && cd build && \
    cmake \
        -D USE_CUDNN=0 \
        -D USE_NCCL=0 \
        -D BLAS=open \
        -D USE_OPENCV=ON .. && \
    make -j"$(nproc)" all && \
    make pycaffe && \
    make test && \ 
    make install && \
    echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig && \
    rm -rf $CAFFE_ROOT

RUN cd /src && \
    mkdir -p $OPENCV_DIR && \
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip && \
    unzip 3.3.1.zip && \ 
    mv /src/opencv-$OPENCV_VERSION/ $OPENCV_DIR/ && \ 
    rm -rf /src/$OPENCV_VERSION.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip -O $OPENCV_VERSION-contrib.zip && \
    unzip $OPENCV_VERSION-contrib.zip && \
    mv /src/opencv_contrib-$OPENCV_VERSION $OPENCV_DIR/ && \ 
    rm -rf /src/$OPENCV_VERSION-contrib.zip && \ 
    mkdir -p $OPENCV_DIR/opencv-$OPENCV_VERSION/build && \
    cd $OPENCV_DIR/opencv-$OPENCV_VERSION/build && \ 
    cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D BUILD_PYTHON_SUPPORT=ON \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D CUDA_CUDA_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D BUILD_PYTHON_SUPPORT=ON \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
    -D PYTHON_INCLUDE_DIR=/usr/include/python3.5 \
    -D PYTHON_INCLUDE_DIR2=/usr/include/x86_64-linux-gnu/python3.5m \
    -D PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.5/dist-packages/numpy/core/include/ \
    -D OPENCV_EXTRA_MODULES_PATH=$OPENCV_DIR/opencv_contrib-$OPENCV_VERSION/modules \
    -D WITH_TBB=ON \
    -D WITH_PTHREADS_PF=ON \
    -D WITH_OPENNI=OFF \
    -D WITH_OPENNI2=ON \
    -D WITH_EIGEN=ON \
    -D BUILD_DOCS=ON \
    -D BUILD_TESTS=ON \
    -D BUILD_PERF_TESTS=ON \
    -D BUILD_EXAMPLES=ON \
    -D WITH_OPENCL=$OPENCL_ENABLED \
    -D WITH_CUDA=$CUDA_ENABLED \
    -D WITH_CUBLAS=$CUDA_ENABLED \
    -D USE_GStreamer=ON \
    -D WITH_GDAL=ON \
    -D WITH_CSTRIPES=ON \
    -D WITH_NVCUVID=$CUDA_ENABLED \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=$CUDA_ENABLED \
    -D BUILD_opencv_gpu=$CUDA_ENABLED \
    -D BUILD_opencv_gpuarithm=$CUDA_ENABLED \
    -D BUILD_opencv_gpubgsegm=$CUDA_ENABLED \
    -D BUILD_opencv_gpucodec=$CUDA_ENABLED \
    -D BUILD_opencv_gpufeatures2d=$CUDA_ENABLED \
    -D BUILD_opencv_gpufilters=$CUDA_ENABLED \
    -D BUILD_opencv_gpuimgproc=$CUDA_ENABLED \
    -D BUILD_opencv_gpulegacy=$CUDA_ENABLED \
    -D BUILD_opencv_gpuoptflow=$CUDA_ENABLED \
    -D BUILD_opencv_gpustereo=$CUDA_ENABLED \
    -D BUILD_opencv_gpuwarping=$CUDA_ENABLED \
    -D WITH_OPENGL=ON \
    -D WITH_QT=OFF \
    -D WITH_IPP=ON \
    -D WITH_V4L=ON .. && \
    make -j8 && \
    make install && \
    ldconfig && \
    rm -rf $OPENCV_DIR

RUN cd /home/ubuntu && \
    pip3 install virtualenv && \
    virtualenv -p python3 ml_venv && \
    . ml_venv/bin/activate && \
    ln -s /usr/local/lib/python3.5/dist-packages/cv2.cpython-35m-x86_64-linux-gnu.so /home/ubuntu/ml_venv/lib/python3.5/site-packages/ && \
    pip3 install --upgrade --force-reinstall numpy \
                 protobuf \
		 pandas \
                 Pillow \
                 future \
                 graphviz \
                 joblib \
                 matplotlib \
                 theano \
                 cython \
                 scikit-learn \
                 scikit-image \
                 scipy \
                 tensorflow-gpu \
                 tensorflow-tensorboard \
                 tflearn \
                 keras \
                 jupyter \
                 sphinx \
                 wheel \
                 boost \
                 pytest \
		 librosa \
		 grpcio \
		 flask \
		 flask-restplus \
                 http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl && \
    pip3 install torchvision && \
    pip3 install git+https://github.com/hlin117/mdlp-discretization

RUN mkdir $LIBGPUARRAY_DIR/ && \
	cd /src && \
	pip3 install Mako && \
	git clone https://github.com/Theano/libgpuarray.git && \
	mv libgpuarray $LIBGPUARRAY_DIR/ && \
	cd $LIBGPUARRAY_DIR/libgpuarray && \
	mkdir Build && \
	cd Build && \
	cmake \
	-D CMAKE_BUILD_TYPE=Release .. && \
	make && \
	make install && \
	rm -rf $LIBGPUARRAY_DIR && \
	cd /

USER $NB_USER

EXPOSE 6006 

WORKDIR /

CMD ["/bin/bash"]
