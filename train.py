import sys
import time
from multiprocessing import Process, Queue

import yaml
import numpy as np
import zmq
import pycuda.driver as drv

sys.path.append('./lib')
from tools import (save_weights, load_weights,
                   save_momentums, load_momentums)
from train_funcs import (unpack_configs, adjust_learning_rate,
                         get_val_error_loss, get_rand3d, train_model_wrap,
                         proc_configs)


def train_net(config):

    # UNPACK CONFIGS
    (flag_para_load, train_filenames, val_filenames,
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

    ## BUILD NETWORK ##
    model = AlexNet(config)
    layers = model.layers
    batch_size = model.batch_size

    ## COMPILE FUNCTIONS ##
    (train_model, validate_model, train_error, learning_rate,
        shared_x, shared_y, rand_arr, vels) = compile_models(model, config)


    ######################### TRAIN MODEL ################################

    print '... training'

    if flag_para_load:
        # pass ipc handle and related information
        gpuarray_batch = theano.misc.pycuda_utils.to_gpuarray(
            shared_x.container.value)
        h = drv.mem_get_ipc_handle(gpuarray_batch.ptr)
        sock.send_pyobj((gpuarray_batch.shape, gpuarray_batch.dtype, h))

        load_send_queue.put(img_mean)

    n_train_batches = len(train_filenames)
    minibatch_range = range(n_train_batches)



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
                                       flag_para_load,
                                       config['batch_crop_mirror'],###BUG 1 FIXED#####
                                       send_queue=load_send_queue,
                                       recv_queue=load_recv_queue)


            if num_iter % config['print_freq'] == 0:
                print 'training @ iter = ', num_iter
                print 'training cost:', cost_ij
                if config['print_train_error']:
                    print 'training error rate:', train_error()

            if flag_para_load and (count < len(minibatch_range)):
                load_send_queue.put('calc_finished')

        ############### Test on Validation Set ##################

        DropoutLayer.SetDropoutOff()

        this_validation_error, this_validation_loss = get_val_error_loss(
            rand_arr, shared_x, shared_y,
            val_filenames, val_labels,
            flag_para_load, img_mean,
            batch_size, validate_model,
            send_queue=load_send_queue, recv_queue=load_recv_queue)


        print('epoch %i: validation loss %f ' %
              (epoch, this_validation_loss))
        print('epoch %i: validation error %f %%' %
              (epoch, this_validation_error * 100.))
        val_record.append([this_validation_error, this_validation_loss])
        np.save(config['weights_dir'] + 'val_record.npy', val_record)

        DropoutLayer.SetDropoutOn()
        ############################################

        # Adapt Learning Rate
        step_idx = adjust_learning_rate(config, epoch, step_idx,
                                        val_record, learning_rate)

        # Save weights
        if epoch % config['snapshot_freq'] == 0:
            save_weights(layers, config['weights_dir'], epoch)
            np.save(config['weights_dir'] + 'lr_' + str(epoch) + '.npy',
                       learning_rate.get_value())
            save_momentums(vels, config['weights_dir'], epoch)

    print('Optimization complete.')


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    with open('spec_1gpu.yaml', 'r') as f:
        config = dict(config.items() + yaml.load(f).items())
        
    config = proc_configs(config)






    if config['para_load']:
        from proc_load import fun_load
        config['queue_l2t'] = Queue(1)
        config['queue_t2l'] = Queue(1)
        train_proc = Process(target=train_net, args=(config,))
        load_proc = Process(
            target=fun_load, args=(config, config['sock_data']))
        train_proc.start()
        load_proc.start()
        train_proc.join()
        load_proc.join()

    else:
        train_proc = Process(target=train_net, args=(config,))
        train_proc.start()
        train_proc.join()
