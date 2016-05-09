# What is in this repo?
A re-implementation of [Asynchronous Methods for Deep Reinforcement Learning]
(https://arxiv.org/abs/1602.01783) using [TensorFlow r0.7]
(https://www.tensorflow.org/).

[![Gitter](https://badges.gitter.im/traai/async-deep-rl.svg)](https://gitter.im/traai/async-deep-rl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


# How to run the algorithms (MacOSX for now)?
(1) Clone this repo at `~/some-path`.

(2) Make sure your machine has docker installed. Follow instructions [here]
(https://docs.docker.com/engine/installation/#install-docker-engine) if not.

(3) Make sure you have xquartz installed in order to visualise game play. 
Do the following in a separate terminal window:
```
$ brew cask install --force xquartz
$ open -a XQuartz
$ socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
```

(4) Get our docker image containing all dependencies to run the algorithms and 
to visualise game play.
```shell
$ docker pull restrd/tensorflow-atari-cpu
```

(5) Run the docker image. This will mount your home folder to `/your-user-name` 
inside the container. Be sure to give a name to the container: 
`<container-name>`
```shell
$ docker run -d -p 8888:8888 -p 6006:6006 --name <container-name> -v ~/:/root/$usr -e DISPLAY=$(ifconfig vboxnet0 | awk '$1 == "inet" {gsub(/\/.*$/, "", $2); print $2}'):0 -it docker.io/restrd/tensorflow-atari-cpu
```

(6) Shell into the container.
```
$ docker exec -it <container-name> /bin/bash
```

(7) Go to the algorithms folder 
(`/your-user-name/some-path/async-deep-rl/algorithms`) and choose which 
algorithm to run via the configuration options in `main.py`.

(8) Run the algorithms, e.g.:
```shell
$ python main.py beam_rider ../atari_roms/ 1 &
```

# Running [TensorBoard](https://www.tensorflow.org/versions/r0.8/how_tos/summaries_and_tensorboard/index.html)
You can also run [TensorBoard](https://www.tensorflow.org/versions/r0.8/how_tos/summaries_and_tensorboard/index.html) 
to visualise losses and game scores. 

(1) Run tensorboard from within the container:
```
$ tensorboard --logdir=/tmp/summary_logs/ &
```

(2) Get the ip address of your docker host running inside of [VirtualBox]
(https://www.virtualbox.org/). Go to `http://<docker-host-ip>:6006`


# Convergence issues (May 9, 2016)
The implementation is still in flux. We are still trying to get these algorithms 
to converge. We are using Python threading, which may (or may not?) be 
disrupting the Hogwild!ness of the algorithms.
