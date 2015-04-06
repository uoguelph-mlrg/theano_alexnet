'''
For loading the weights and test on a single GPU to make sure the measured performance is correct
'''
import sys
from multiprocessing import Process, Queue

import yaml
import zmq
import pycuda.driver as drv

sys.path.append('./lib')
from tools import load_weights
from train_funcs import (unpack_configs, get_val_error_loss, 
                         proc_configs)


def validate_performance(config):

    # UNPACK CONFIGS
    (flag_para_load, flag_datalayer, train_filenames, val_filenames,
     train_labels, val_labels, img_mean) = unpack_configs(config)

    if flag_para_load:
        # pycuda and zmq set up
        drv.init()
        dev = drv.Device(int(config['gpu'][-1]))
        ctx = dev.make_context()
        sock = zmq.Context().socket(zmq.PAIR)
        sock.connect('tcp://localhost:{0}'.format(config['sock_data']))

        load_send_queue = config['queue_t2l']
        load_recv_queue = config['queue_l2t']
    else:
        load_send_queue = None
        load_recv_queue = None

    import theano.sandbox.cuda
    theano.sandbox.cuda.use(config['gpu'])
    import theano
    theano.config.on_unused_input = 'warn'

    from layers import DropoutLayer
    from alex_net import AlexNet, compile_models

    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils

    # # BUILD NETWORK ##
    model = AlexNet(config)
    layers = model.layers
    batch_size = model.batch_size

    # # COMPILE FUNCTIONS ##
    (train_model, validate_model, train_error, learning_rate,
        shared_x, shared_y, rand_arr, vels) = compile_models(model, config,
                                                             flag_top_5=True)

    print '... training'

    if flag_para_load:
        # pass ipc handle and related information
        gpuarray_batch = theano.misc.pycuda_utils.to_gpuarray(
            shared_x.container.value)
        h = drv.mem_get_ipc_handle(gpuarray_batch.ptr)
        sock.send_pyobj((gpuarray_batch.shape, gpuarray_batch.dtype, h))

    load_epoch = config['load_epoch']
    load_weights(layers, config['weights_dir'], load_epoch)

    DropoutLayer.SetDropoutOff()

    load_send_queue.put(img_mean)
    this_validation_error, this_validation_error_top_5, this_validation_loss = \
        get_val_error_loss(rand_arr, shared_x, shared_y,
                           val_filenames, val_labels,
                           flag_datalayer, flag_para_load,
                           batch_size, validate_model,
                           send_queue=load_send_queue,
                           recv_queue=load_recv_queue,
                           flag_top_5=True)

    print('validation error %f %%' %
          (this_validation_error * 100.))
    print('top 5 validation error %f %%' %
          (this_validation_error_top_5 * 100.))
    print('validation loss %f ' %
          (this_validation_loss))

    return this_validation_error, this_validation_loss

    ############################################


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    with open('spec_1gpu.yaml', 'r') as f:
        config = dict(config.items() + yaml.load(f).items())

    config = proc_configs(config)

    config['resume_train'] = True
    config['load_epoch'] = 65

    if config['para_load']:
        from proc_load import fun_load
        config['queue_l2t'] = Queue(1)
        config['queue_t2l'] = Queue(1)
        train_proc = Process(target=validate_performance, args=(config,))
        load_proc = Process(
            target=fun_load, args=(config, config['sock_data']))
        train_proc.start()
        load_proc.start()
        train_proc.join()
        load_proc.join()

    else:
        train_proc = Process(target=validate_performance, args=(config,))
        train_proc.start()
        train_proc.join()
