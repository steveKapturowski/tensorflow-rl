# What is in this repo?

[![Join the chat at https://gitter.im/traai/async-deep-rl](https://badges.gitter.im/traai/async-deep-rl.svg)](https://gitter.im/traai/async-deep-rl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A [Tensorflow](https://www.tensorflow.org/)-based implementation of all algorithms presented in [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783).

This implementation uses processes instead of threads to achieve real concurrency. Each process has a local replica of the network(s) used, implemented in Tensorflow, and runs its own Tensorflow session. In addition, a copy of the network parameters are kept in a shared memory space. At runtime, each process uses its own local network(s) to choose actions and compute gradients (with Tensorflow). The shared network parameters are updated periodically in an asynchronous manner, by applying the grads obtained from Tensorflow into the shared memory space. 

Both [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment) and [Open AI GYM](https://gym.openai.com/) environments can be used.


The graphs below show the reward achieved in different games by one individual actor during training (i.e., not averaging over several runs, and over all actors, as in the paper). All experiments were run on a rather old machine equipped with 2 Xeon E5540 quad-core 2.53GHz CPUs (16 virtual cores) and 47 Gb RAM.

Boxing-v0 (from OpenAI Gym), A3C, 100 actors, lr=0.0007, 80M steps in 59h, 31m:
![](https://github.com/traai/async-deep-rl/blob/master/help/images/boxing_v0.png)
As you can see, the score achieved is much higher than the one reported in the paper (and maybe even state-of-the-art? a proper evaluation should be needed). That is due to the effect of having 100 actors. So concurrently exploring the environment in different ways definitely helps with the learning process and makes experience replay not needed. Note, however, that the performance in terms of training time is slightly worse than with fewer actors. This is probably due to our implementation, which is not optimal, and to the limitations of the machine we used.
Pong (from ALE), A3C, 16 actors, lr=0.0007, 80M steps in 48h:
![](https://github.com/traai/async-deep-rl/blob/master/help/images/pong.png)

Beam Rider (from ALE), A3C, 16 actors, lr=0.0007, 80M steps in 45h, 25min:
![](https://github.com/traai/async-deep-rl/blob/master/help/images/beamrider.png)

Breakout (from ALE), A3C, 15 actors, lr=0.0007, 80M steps in 53h, 22m:
![](https://github.com/traai/async-deep-rl/blob/master/help/images/breakout.png)



# How to run the algorithms (MacOSX for now)?
A number of hyperparameters can be specified. Default values have been chosen according to the paper and [information](https://github.com/muupan/async-rl/wiki) received by @muupan from the authors. To see a list, please run:
```
python main.py -h
```

If you just want to see the code in action, you can kick off training with the default hyperparameters by running:
```
python main.py pong --rom_path ../atari_roms/
```

To run outside a docker, you need to install some dependencies:
- Tensorflow
- [OpenAI Gym](https://github.com/openai/gym#installation)
- The Arcade Learning Environment ([ALE](https://github.com/mgbellemare/Arcade-Learning-Environment)).(Note that OpenAI Gym uses ALE internally, so you could use that version. This would require some hacking.)
- Scikit-image
- Open CV v2, for standalone ALE (It should be possible to change the code in `emulator.py` to use Scikit-image instead of CV2. Indeed, CV2 might slow things down)  

To run inside a docker:
(1) Clone this repo at `~/some-path`.

(2) Make sure your machine has docker installed. Follow instructions [here]
(https://docs.docker.com/toolbox/toolbox_install_mac/) if not. [These] 
(https://docs.docker.com/toolbox/toolbox_install_windows/) instructions may work for Windows.

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
$ docker run -d -p 8888:8888 -p 6006:6006 --name "<container-name>" -v ~/:/root/$usr -e DISPLAY=$(ifconfig vboxnet0 | awk '$1 == "inet" {gsub(/\/.*$/, "", $2); print $2}'):0 -it docker.io/restrd/tensorflow0.10-atari-cpu
```

(6) Shell into the container.
```
$ docker exec -it <container-name> /bin/bash
```

(7) Go to the algorithms folder 
(`/your-user-name/some-path/async-deep-rl/algorithms`) and choose which 
algorithm to run via the configuration options in `main.py`.

(8) If you want to run the algorithms using [Open AI GYM](https://gym.openai.com/) with 16 processes and visualize the games, e.g.:
```shell
$ python main.py BeamRider-v0 --env GYM -n 16 -v 1 
```

# Running [TensorBoard](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html)
You can also run [TensorBoard](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html) 
to visualise losses and game scores. 

(1) Configure port forwarding rules in [VirtualBox]
(https://www.virtualbox.org/). Go to your running virtual machine's `Settings>Network>Port Forwarding`, and add a new rule (see row starting with tb in pic).

![Setting port forwarding in VirtualBox](https://github.com/traai/async-deep-rl/blob/master/help/images/tb.png)

(2) Run tensorboard from within the container:
```
$ tensorboard --logdir=/tmp/summary_logs/ &
```

(3) If not (1), get the ip address of your docker host running inside of [VirtualBox]
(https://www.virtualbox.org/). Go to `http://<docker-host-ip>:6006`

If (1), go to `http://127.0.0.1:6006`