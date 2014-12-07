'''
Train AlexNet on ImageNet with 2 GPUs.
'''

import sys
import time
from multiprocessing import Process, Queue

import yaml
import numpy as np
import zmq
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

sys.path.append('./lib')
from tools import (save_weights, load_weights,
                   save_momentums, load_momentums)
from train_funcs import (unpack_configs, adjust_learning_rate,
                         get_val_error_loss, get_rand3d, train_model_wrap,
                         proc_configs)


def train_net(config, private_config):

    # UNPACK CONFIGS
    (flag_para_load, flag_datalayer, train_filenames, val_filenames,
     train_labels, val_labels, img_mean) = \
        unpack_configs(config, ext_data=private_config['ext_data'],
                       ext_label=private_config['ext_label'])


    gpu_send_queue = private_config['queue_gpu_send']
    gpu_recv_queue = private_config['queue_gpu_recv']

    # pycuda and zmq set up
    drv.init()
    dev = drv.Device(int(private_config['gpu'][-1]))
    ctx = dev.make_context()

    sock_gpu = zmq.Context().socket(zmq.PAIR)
    if private_config['flag_client']:
        sock_gpu.connect('tcp://localhost:{0}'.format(config['sock_gpu']))
    else:
        sock_gpu.bind('tcp://*:{0}'.format(config['sock_gpu']))

    if flag_para_load:
        sock_data = zmq.Context().socket(zmq.PAIR)
        sock_data.connect('tcp://localhost:{0}'.format(
            private_config['sock_data']))

        load_send_queue = private_config['queue_t2l']
        load_recv_queue = private_config['queue_l2t']
    else:
        load_send_queue = None
        load_recv_queue = None

    import theano.sandbox.cuda
    theano.sandbox.cuda.use(private_config['gpu'])
    import theano
    theano.config.on_unused_input = 'warn'

    from layers import DropoutLayer
    from alex_net import AlexNet, compile_models

    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils

    ## BUILD NETWORK ##
    model = AlexNet(config)
    layers = model.layers
    batch_size = model.batch_size

    ## COMPILE FUNCTIONS ##
    (train_model, validate_model, train_error, learning_rate,
     shared_x, shared_y, rand_arr, vels) = compile_models(model, config)

    total_params = model.params + vels
    # total_params = model.params

    # initialize gpuarrays that points to the theano shared variable
    # pass parameters and other stuff
    param_ga_list = []
    param_other_list = []
    param_ga_other_list = []
    h_list = []
    shape_list = []
    dtype_list = []
    average_fun_list = []

    for param in total_params:
        param_other = theano.shared(param.get_value())
        param_ga = \
            theano.misc.pycuda_utils.to_gpuarray(param.container.value)
        param_ga_other = \
            theano.misc.pycuda_utils.to_gpuarray(
                param_other.container.value)
        h = drv.mem_get_ipc_handle(param_ga.ptr)
        average_fun = \
            theano.function([], updates=[(param,
                                          (param + param_other) / 2.)])

        param_other_list.append(param_other)
        param_ga_list.append(param_ga)
        param_ga_other_list.append(param_ga_other)
        h_list.append(h)
        shape_list.append(param_ga.shape)
        dtype_list.append(param_ga.dtype)
        average_fun_list.append(average_fun)

    # pass shape, dtype and handles
    sock_gpu.send_pyobj((shape_list, dtype_list, h_list))
    shape_other_list, dtype_other_list, h_other_list = sock_gpu.recv_pyobj()

    param_ga_remote_list = []

    # create gpuarray point to the other gpu use the passed information
    for shape_other, dtype_other, h_other in zip(shape_other_list,
                                                 dtype_other_list,
                                                 h_other_list):
        param_ga_remote = \
            gpuarray.GPUArray(shape_other, dtype_other,
                              gpudata=drv.IPCMemoryHandle(h_other))

        param_ga_remote_list.append(param_ga_remote)

    print "Information passed between 2 GPUs"

    ##########################################

    ######################### TRAIN MODEL ################################

    print '... training'

    if flag_para_load:
        # pass ipc handle and related information
        gpuarray_batch = theano.misc.pycuda_utils.to_gpuarray(
            shared_x.container.value)
        h = drv.mem_get_ipc_handle(gpuarray_batch.ptr)
        sock_data.send_pyobj((gpuarray_batch.shape, gpuarray_batch.dtype, h))

        load_send_queue.put(img_mean)

    n_train_batches = len(train_filenames)
    minibatch_range = range(n_train_batches)


    # gpu sync before start
    gpu_send_queue.put('before_start')
    assert gpu_recv_queue.get() == 'before_start'

    # Start Training Loop
    epoch = 0
    step_idx = 0
    val_record = []
    while epoch < config['n_epochs']:
        epoch = epoch + 1

        if config['shuffle']:
            np.random.shuffle(minibatch_range)

        if config['resume_train'] and epoch == 1:
            load_epoch = config['load_epoch']
            load_weights(layers, config['weights_dir'], load_epoch)
            epoch = load_epoch + 1
            lr_to_load = np.load(
                config['weights_dir'] + 'lr_' + str(load_epoch) + '.npy')
            val_record = list(
                np.load(config['weights_dir'] + 'val_record.npy'))
            learning_rate.set_value(lr_to_load)
            load_momentums(vels, config['weights_dir'], epoch)

        if flag_para_load:
            # send the initial message to load data, before each epoch
            load_send_queue.put(str(train_filenames[minibatch_range[0]]))
            if not flag_datalayer:
                load_send_queue.put(get_rand3d())

            # clear the sync before 1st calc
            load_send_queue.put('calc_finished')

        count = 0
        for minibatch_index in minibatch_range:

            num_iter = (epoch - 1) * n_train_batches + count
            count = count + 1
            if count == 1:
                s = time.time()
            if count == 20:
                e = time.time()
                print "time per 20 iter:", (e - s)

            cost_ij = train_model_wrap(train_model, shared_x,
                                       shared_y, rand_arr, img_mean,
                                       count, minibatch_index,
                                       minibatch_range, batch_size,
                                       train_filenames, train_labels,
                                       flag_para_load, flag_datalayer,
                                       send_queue=load_send_queue,
                                       recv_queue=load_recv_queue)

            # gpu sync
            gpu_send_queue.put('after_train')
            assert gpu_recv_queue.get() == 'after_train'

            # exchanging weights
            for param_ga, param_ga_other, param_ga_remote in \
                    zip(param_ga_list, param_ga_other_list,
                        param_ga_remote_list):

                drv.memcpy_peer(param_ga_other.ptr,
                                param_ga_remote.ptr,
                                param_ga_remote.dtype.itemsize *
                                param_ga_remote.size,
                                ctx, ctx)

            ctx.synchronize()

            # gpu sync
            gpu_send_queue.put('after_ctx_sync')
            assert gpu_recv_queue.get() == 'after_ctx_sync'

            # do average
            for average_fun in average_fun_list:
                average_fun()


            # report train stats
            if num_iter % config['print_freq'] == 0:

                gpu_send_queue.put(cost_ij)
                that_cost = gpu_recv_queue.get()
                cost_ij = (cost_ij + that_cost) / 2.

                if private_config['flag_verbose']:
                    print 'training @ iter = ', num_iter
                    print 'training cost:', cost_ij

                if config['print_train_error']:
                    error_ij = train_error()

                    gpu_send_queue.put(error_ij)
                    that_error = gpu_recv_queue.get()
                    error_ij = (error_ij + that_error) / 2.

                    if private_config['flag_verbose']:
                        print 'training error rate:', error_ij

            if flag_para_load and (count < len(minibatch_range)):
                load_send_queue.put('calc_finished')

        ############### Test on Validation Set ##################

        DropoutLayer.SetDropoutOff()

        this_val_error, this_val_loss = get_val_error_loss(
            rand_arr, shared_x, shared_y,
            val_filenames, val_labels,
            flag_datalayer, flag_para_load,
            batch_size, validate_model,
            send_queue=load_send_queue, recv_queue=load_recv_queue)

        # report validation stats
        gpu_send_queue.put(this_val_error)
        that_val_error = gpu_recv_queue.get()
        this_val_error = (this_val_error + that_val_error) / 2.

        gpu_send_queue.put(this_val_loss)
        that_val_loss = gpu_recv_queue.get()
        this_val_loss = (this_val_loss + that_val_loss) / 2.

        if private_config['flag_verbose']:
            print('epoch %i: validation loss %f ' %
                  (epoch, this_val_loss))
            print('epoch %i: validation error %f %%' %
                  (epoch, this_val_error * 100.))
        val_record.append([this_val_error, this_val_loss])

        if private_config['flag_save']:
            np.save(config['weights_dir'] + 'val_record.npy', val_record)

        DropoutLayer.SetDropoutOn()
        ############################################

        # Adapt Learning Rate
        step_idx = adjust_learning_rate(config, epoch, step_idx,
                                        val_record, learning_rate)

        # Save Weights, only one of them will do
        if private_config['flag_save']:
            if epoch % config['snapshot_freq'] == 0:
                save_weights(layers, config['weights_dir'], epoch)
                np.save(config['weights_dir'] + 'lr_' + str(epoch) + '.npy',
                        learning_rate.get_value())
                save_momentums(vels, config['weights_dir'], epoch)

    print('Optimization complete.')


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    with open('spec_2gpu.yaml', 'r') as f:
        config = dict(config.items() + yaml.load(f).items())

    config = proc_configs(config)





    queue_gpu_0to1 = Queue(1)
    queue_gpu_1to0 = Queue(1)

    private_config_0 = {}
    private_config_0['queue_gpu_send'] = queue_gpu_0to1
    private_config_0['queue_gpu_recv'] = queue_gpu_1to0
    private_config_0['sock_data'] = config['sock_data0']
    private_config_0['gpu'] = config['gpu0']
    private_config_0['ext_data'] = '.hkl'
    private_config_0['ext_label'] = '.npy'
    private_config_0['ext_data'] = '_0.hkl'
    private_config_0['ext_label'] = '_0.npy'
    private_config_0['flag_client'] = True
    private_config_0['flag_verbose'] = True
    private_config_0['flag_save'] = True

    private_config_1 = {}
    private_config_1['queue_gpu_send'] = queue_gpu_1to0
    private_config_1['queue_gpu_recv'] = queue_gpu_0to1
    private_config_1['sock_data'] = config['sock_data1']
    private_config_1['gpu'] = config['gpu1']
    private_config_1['ext_data'] = '_1.hkl'
    private_config_1['ext_label'] = '_1.npy'
    private_config_1['flag_client'] = False
    private_config_1['flag_verbose'] = False
    private_config_1['flag_save'] = False



    if config['para_load']:
        from proc_load import fun_load
        private_config_0['queue_l2t'] = Queue(1)
        private_config_0['queue_t2l'] = Queue(1)
        train_proc_0 = Process(target=train_net,
                               args=(config, private_config_0))
        load_proc_0 = Process(target=fun_load,
                              args=(dict(private_config_0.items() +
                                         config.items()),
                                    private_config_0['sock_data']))

        private_config_1['queue_l2t'] = Queue(1)
        private_config_1['queue_t2l'] = Queue(1)
        train_proc_1 = Process(target=train_net,
                               args=(config, private_config_1))
        load_proc_1 = Process(target=fun_load,
                              args=(dict(private_config_1.items() +
                                         config.items()),
                                    private_config_1['sock_data']))

        train_proc_0.start()
        load_proc_0.start()
        train_proc_1.start()
        load_proc_1.start()

        train_proc_0.join()
        load_proc_0.join()
        train_proc_1.join()
        load_proc_1.join()

    else:
        train_proc_0 = Process(target=train_net,
                               args=(config, private_config_0))
        train_proc_1 = Process(target=train_net,
                               args=(config, private_config_1))
        train_proc_0.start()
        train_proc_1.start()
        train_proc_0.join()
        train_proc_1.join()
