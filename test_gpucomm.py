import sys,os
import time
from multiprocessing import Process, Queue

import yaml
import numpy as np
import zmq
# import pycuda.gpuarray as gpuarray

from lib.tools import (save_weights, load_weights,
                   save_momentums, load_momentums)
from train_funcs import (unpack_configs, adjust_learning_rate,
                         get_val_error_loss, get_rand3d, train_model_wrap,
                         proc_configs)
                         
def get_intranode_comm(ctx, local_size, local_rank, seed_str='fakeID'):
    
    '''a gpucomm between all synchronous workers within a node'''
    
    from pygpu import collectives

    local_id = collectives.GpuCommCliqueId(context=ctx)

    # string =  local_id.comm_id.decode('utf-8')
    # for value in local_id.comm_id:
    #     print(value)
    # for value in string:
    #     print(value)
    
    #============reproduce ID====================
    s = ('nccl-%s-0' % os.getpid()).encode('utf-8')
    ba=bytearray(128) 
    for ind,_ in enumerate(s):
        ba[ind]=s[ind]
    # string = ba.decode('utf-8')
    # print(type(ba),type(string))
    # print(len(ba),len(string))
    # print(ba,string,os.getpid())
    # print(list(ba),list(string))
    assert ba==local_id

    #============fake ID====================
    s = ('nccl-%s-0' % seed_str).encode('utf-8')
    
    ba=bytearray(128) 
    for ind,_ in enumerate(s):
        ba[ind]=s[ind]
    
    local_id.comm_id = ba

    gpucomm = collectives.GpuComm(local_id,local_size,local_rank)

    return gpucomm
    
def train_net(config, private_config):
    
    import os
    if 'THEANO_FLAGS' in os.environ:
        raise ValueError('Use theanorc to set the theano config')
    os.environ['THEANO_FLAGS'] = 'device={0}'.format(private_config['gpu'])
    import theano.gpuarray
    # This is a bit of black magic that may stop working in future
    # theano releases
    ctx = theano.gpuarray.type.get_context(None)
            
    
    gpucomm=get_intranode_comm(ctx, 2 ,int(private_config['gpu'][-1]))
    
    
if __name__ == '__main__':
    
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    with open('spec_2gpu.yaml', 'r') as f:
        # config = {**config,**yaml.load(f)}
        config = dict(config.items() + yaml.load(f).items())

    config = proc_configs(config)
    
    private_config_0 = {}
    private_config_0['gpu'] = config['gpu0']
    private_config_0['ext_data'] = '.hkl'
    private_config_0['ext_label'] = '.npy'
    private_config_0['ext_data'] = '_0.hkl'
    private_config_0['ext_label'] = '_0.npy'
    private_config_0['flag_client'] = True
    private_config_0['flag_verbose'] = True
    private_config_0['flag_save'] = True

    private_config_1 = {}
    private_config_1['gpu'] = config['gpu1']
    private_config_1['ext_data'] = '_1.hkl'
    private_config_1['ext_label'] = '_1.npy'
    private_config_1['flag_client'] = False
    private_config_1['flag_verbose'] = False
    private_config_1['flag_save'] = False
    
    train_proc_0 = Process(target=train_net,
                           args=(config, private_config_0))
    train_proc_1 = Process(target=train_net,
                           args=(config, private_config_1))
    train_proc_0.start()
    train_proc_1.start()
    train_proc_0.join()
    train_proc_1.join()