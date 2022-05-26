# FROM ros:melodic-ros-core-bionic
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# ENV LD_LIBRARY_PATH /usr/local/cuda-10.2/lib64:/usr/local/cuda-10.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV HOME /root

#####################
# ROS CORE
#####################
# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO melodic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-ros-core=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*


#####################
# ROS BASE & APPLICATION SPECIFIC PACKAGES
#####################
# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
    rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-ros-base=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

# APPLICATION SPECIFIC ROS PACKAGES
RUN apt-get update && apt-get install -y \
    ros-melodic-pybind11-catkin \
    ros-melodic-moveit \
    ros-melodic-pr2-simulator \
    ros-melodic-moveit-pr2 \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc


#####################
# INSTALL CONDA
#####################
RUN apt-get update --fix-missing && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH


######################
## PYTORCH FROM SOURCE
######################
ENV ENV_NAME=base
#ENV TORCH_VERSION=v1.6.0-rc1
## FOR SOME REASON TORCH FAILS ("NO CUDA DRIVERS ...) IF CHANGING THE PYTHON VERSION HERE!
## not sure how much the above stuff depends on the same version / whether I have to or can define it further above
##ENV PYTHON_VERSION=3.6
#
##RUN conda install -n ${ENV_NAME} python=${PYTHON_VERSION} numpy ninja pyyaml mkl mkl-include setuptools cmake cffi \
#RUN conda install -n ${ENV_NAME} numpy ninja pyyaml mkl mkl-include setuptools cmake cffi \
#    && conda install -n ${ENV_NAME} -c pytorch magma-cuda102 \
#    && conda clean -afy
#
#RUN git clone --recursive --branch ${TORCH_VERSION} https://github.com/pytorch/pytorch \
#    && cd pytorch \
#    && git submodule sync \
#    && git submodule update --init --recursive \
#    && TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
#       CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
#       conda run -n ${ENV_NAME} python setup.py install \
#    && cd .. \
#    && rm -rf pytorch/ \
#    && conda install -n ${ENV_NAME} cudatoolkit=10.2 -c pytorch \
#    && conda clean -afy


#####################
# CONDA DEPENDENCIES FROM ENV.YAML
#####################
# to use git, solve key issue: https://vsupalov.com/build-docker-image-clone-private-repo-ssh-key/
# only want to get the environment.yml at this point, so as not to recreate everytime some code changes
COPY environment.yml src/dllab_modulation_rl/
RUN conda env update -n ${ENV_NAME} -f src/dllab_modulation_rl/environment.yml \
    && conda clean -afy


#######################
### CREATE CATKIN WORKSPACE WITH PYTHON3
#######################
### python3: https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674
RUN conda install -n ${ENV_NAME} -c conda-forge rospkg catkin_pkg \
    && apt-get update \
    && apt-get install -y python-catkin-tools python3-dev python3-numpy python-defusedxml \
    && rm -rf /var/lib/apt/lists/*

# bad practice to pip install ros packages, but somehow the above doesn't do the trick
RUN pip install defusedxml

RUN mkdir -p $HOME/catkin_build_ws/src
WORKDIR $HOME/catkin_build_ws
#RUN catkin config -DPYTHON_EXECUTABLE=/opt/conda/bin/python -DPYTHON_INCLUDE_DIR=/opt/conda/include/python2.7 -DPYTHON_LIBRARY=/opt/conda/lib/libpython2.7.so \
#    && catkin config --install
RUN catkin config \
    && catkin config --install


#####################
# COPY FILES AND BUILD ROS PACKAGE -> don't use conda python, but the original!
#####################
# copy only files required for compilation use caching whenever possible
COPY include/ src/dllab_modulation_rl/include/
COPY src/ src/dllab_modulation_rl/src/
COPY CMakeLists.txt package.xml src/dllab_modulation_rl/
RUN /bin/bash -c ". /opt/ros/melodic/setup.bash && catkin build"
# NEEDS TO BE RUN WITH EVERY RUN IN THE DOCKERFILE: RUN /bin/bash -c ". devel/setup.bash"

COPY ros_startup.sh ros_startup_incl_train.sh ./

# copy all files now
COPY . src/dllab_modulation_rl/


######################
## RUN TRAINING
######################
# Remember to always source the ros setups before running commands in the container, i.e.
#source /opt/ros/melodic/setup.bash
#source devel/setup.bash

## Make RUN commands use the new environment:
#SHELL ["conda", "run", "-n", ${ENV_NAME}, "/bin/bash", "-c"]
#CMD ["conda", "run", "-n", ${ENV_NAME}, "python", "src/dllab_modulation_rl/python/demo.py"]
#CMD bash
CMD bash ros_startup_incl_train.sh "python python/demo.py"

