#AlexNet Implementation with Theano

Demonstration of training an AlexNet in Python with Theano.
Please see this [technical report](http://arxiv.org/abs/1412.2302) for a high level description.
[theano_multi_gpu](https://github.com/uoguelph-mlrg/theano_multi_gpu) provides a toy example on how to use 2 GPUs to train a MLP on the mnist data.

If you use this in your research, we kindly ask that you cite the above report:

```bibtex
@article{ding2014theano,
  title={Theano-based Large-Scale Visual Recognition with Multiple GPUs},
  author={Ding, Weiguang and Wang, Ruoyan and Mao, Fei and Taylor, Graham},
  journal={arXiv preprint arXiv:1412.2302},
  year={2014}
}
```

## Dependencies
* [numpy](http://www.numpy.org/)
* [Theano](http://deeplearning.net/software/theano/) >= 0.10
* <s>[Pylearn2](http://deeplearning.net/software/pylearn2/)</s>
* <s>[PyCUDA](http://mathema.tician.de/software/pycuda/)</s>
* [pygpu/libgpuarray](http://deeplearning.net/software/libgpuarray/installation.html)
* [zeromq](http://zeromq.org/bindings:python)
* [hickle](https://github.com/telegraphic/hickle)

## How to run

### Prepare raw ImageNet data
Download [ImageNet dataset](http://www.image-net.org/download-images) and unzip image files.

### Preprocess the data
This involves shuffling training images, generating data batches, computing the mean image and generating label files.

#### Steps
* Set paths in the preprocessing/paths.yaml. Each path is described in this file. 
* Run preprocessing/generate_data.sh, which will call 3 python scripts and do all the mentioned steps. It runs for about 1~2 days. For a quick trial of the code, run preprocessing/generate_toy_data.sh, which takes ~10 minutes and proceed.

preprocessing/lists.txt is a static file that lists what files should be created by running generate_data.sh.

### Train AlexNet

#### Set configurations
config.yaml contains common configurations for both the 1-GPU and 2-GPU version.

spec_1gpu.yaml and spec_2gpu.yaml contains different configurations for the 1-GPU and 2-GPU version respectively.

If you changed preprocessing/paths.yaml, make sure you change corresponding paths in config.yaml, spec_1gpu.yaml and spec_2gpu.yaml accordingly.
#### Start training

1-GPU version, run:

backend=gpuarray <s>THEANO_FLAGS=mode=FAST_RUN,floatX=float32</s> python train.py

2-GPU version, run:

THEANO_FLAGS=mode=FAST_RUN,floatX=float32 python train_2gpu.py

Validation error and loss values are stored as weights_dir/val_record.npy

Here we do not set device to gpu in THEANO_FLAGS. Instead, users should control which GPU(s) to use in spec_1gpu.yaml and spec_2gpu.yaml.

### Pretrained AlexNet

Pretrained AlexNet weights and configurations can be found at [pretrained/alexnet](https://github.com/uoguelph-mlrg/theano_alexnet/tree/master/pretrained/alexnet)


## Acknowledgement
*Frédéric Bastien*, for providing the example of [Using Multiple GPUs](https://github.com/Theano/Theano/wiki/Using-Multiple-GPUs)

*Lev Givon*, for helping on inter process communication between 2 gpus with PyCUDA, Lev's original script https://gist.github.com/lebedov/6408165

*Guangyu Sun*, for help on debugging the code
