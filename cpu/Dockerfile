FROM ubuntu:16.04
ARG PYTHON_VERSION=3.5
ARG NB_USER=ubuntu 
ENV OPENCV_DIR /opt/opencv 
ENV CAFFE_ROOT /opt/caffe
ENV NUM_CORES 8 
ENV NB_UID 1000 
ARG CLONE_TAG=1.0
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
ENV OPENCV_VERSION=3.4.0

LABEL maintainer "Pierre-Luc Delisle <pierre-luc.delisle.1@ens.etsmtl.ca>"

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER

RUN apt-get update && apt-get install -y --no-install-recommends \
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
    cd /src && \
    mkdir -p $OPENCV_DIR && \
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip && \
    unzip $OPENCV_VERSION.zip && \ 
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
    -D WITH_OPENCL=ON \
    -D WITH_CUDA=OFF \
    -D WITH_CUBLAS=OFF \
    -D USE_GStreamer=ON \
    -D WITH_GDAL=ON \
    -D WITH_CSTRIPES=ON \
    -D WITH_NVCUVID=OFF \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=OFF \
    -D BUILD_opencv_gpu=OFF \
    -D BUILD_opencv_gpuarithm=OFF \
    -D BUILD_opencv_gpubgsegm=OFF \
    -D BUILD_opencv_gpucodec=OFF \
    -D BUILD_opencv_gpufeatures2d=OFF \
    -D BUILD_opencv_gpufilters=OFF \
    -D BUILD_opencv_gpuimgproc=OFF \
    -D BUILD_opencv_gpulegacy=OFF \
    -D BUILD_opencv_gpuoptflow=OFF \
    -D BUILD_opencv_gpustereo=OFF \
    -D BUILD_opencv_gpuwarping=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_QT=OFF \
    -D WITH_IPP=ON \
    -D WITH_V4L=ON .. && \
    make -j8 && \
    make install && \
    ldconfig && \
    rm -rf $OPENCV_DIR

RUN cd /src && \
    mkdir -p $CAFFE_ROOT && \
    cd $CAFFE_ROOT && \
    git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
    pip install --upgrade pip && \
    cd python && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    cd .. && \
    mkdir build && cd build && \
    cmake \
        -D USE_CUDNN=$CUDA_ENABLED \
        -D USE_NCCL=$CUDA_ENABLED \
        -D BLAS=open \
        -D PYTHON_INCLUDE=/usr/bin/python3 \ 
        /usr/lib/python3.5/dist-packages/numpy/core/include \
        -D USE_OPENCV=ON \
        -D OPENCV_VERSION=3 \
        -D WITH_PYTHON_LAYER=ON .. && \
    make -j"$(nproc)" all && \
    make pycaffe && \
    make test && \ 
    make install && \
    echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig && \
    rm -rf $CAFFE_ROOT

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
                 tensorflow \
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
		 nose \
		 flask \
		 flask-restplus \
                 http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl && \
    pip3 install torchvision && \
    pip3 install git+https://github.com/hlin117/mdlp-discretization

USER $NB_USER

EXPOSE 6006

EXPOSE 5000

WORKDIR /

CMD ["/bin/bash"]
