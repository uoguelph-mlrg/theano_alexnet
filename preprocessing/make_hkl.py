# Preprocessing: From JPEG to HKL

import os
import glob
import sys

import yaml
import scipy.misc
import numpy as np

import hickle as hkl


def get_img(img_name, img_size=256, batch_size=256):

    target_shape = (img_size, img_size, 3)
    img = scipy.misc.imread(img_name)  # x*x*3
    assert img.dtype == 'uint8', img_name
    # assert False

    if len(img.shape) == 2:
        img = scipy.misc.imresize(img, (img_size, img_size))
        img = np.asarray([img, img, img])
    else:
        if img.shape[2] > 3:
            img = img[:, :, :3]
        img = scipy.misc.imresize(img, target_shape)
        img = np.rollaxis(img, 2)
    if img.shape[0] != 3:
        print img_name
    return img


def save_batches(file_list, tar_dir, img_size=256, batch_size=256,
                 flag_avg=False, num_sub_batch=1):
    '''
    num_sub_batch is for parallelling using multiple gpus, it should be
    2, 4, or 8,
    where the indexing is reverted binary number
    when 2, the files ends with _0.pkl and _1.pkl
    when 4, with _00.pkl, _10.pkl, _01.pkl and _11.pkl

    '''

    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    img_batch = np.zeros((3, img_size, img_size, batch_size), np.uint8)

    if flag_avg:
        img_sum = np.zeros((3, img_size, img_size))

    batch_count = 0
    count = 0
    for file_name in file_list:
        img_batch[:, :, :, count % batch_size] = \
            get_img(file_name, img_size=img_size, batch_size=batch_size)

        count += 1
        if count % batch_size == 0:
            batch_count += 1

            if flag_avg:
                img_sum += img_batch.mean(axis=3)

            if num_sub_batch == 1:
                save_name = '%04d' % (batch_count - 1) + '.hkl'
                hkl.dump(img_batch, os.path.join(tar_dir, save_name), mode='w')

            elif num_sub_batch == 2:
                half_size = batch_size / 2
                save_name = '%04d' % (batch_count - 1) + '_0.hkl'
                hkl.dump(img_batch[:, :, :, :half_size],
                         os.path.join(tar_dir, save_name), mode='w')

                save_name = '%04d' % (batch_count - 1) + '_1.hkl'
                hkl.dump(img_batch[:, :, :, half_size:],
                         os.path.join(tar_dir, save_name), mode='w')

            elif num_sub_batch == 4:
                q1 = batch_size / 4
                q2 = batch_size / 2
                q3 = batch_size / 4 * 3

                save_name = '%04d' % (batch_count - 1) + '_00.hkl'
                hkl.dump(img_batch[:, :, :, :q1],
                         os.path.join(tar_dir, save_name), mode='w')
                save_name = '%04d' % (batch_count - 1) + '_10.hkl'
                hkl.dump(img_batch[:, :, :, q1:q2],
                         os.path.join(tar_dir, save_name), mode='w')
                save_name = '%04d' % (batch_count - 1) + '_01.hkl'
                hkl.dump(img_batch[:, :, :, q2:q3],
                         os.path.join(tar_dir, save_name), mode='w')
                save_name = '%04d' % (batch_count - 1) + '_11.hkl'
                hkl.dump(img_batch[:, :, :, q3:],
                         os.path.join(tar_dir, save_name), mode='w')
            else:
                NotImplementedError("num_sub_batch has to be 1, 2, or 4")

    return img_sum / batch_count if flag_avg else None


def get_train_filenames(src_train_dir, misc_dir, seed=1):

    if os.path.exists(os.path.join(misc_dir, 'shuffled_train_filenames.npy')):
        return np.load(os.path.join(misc_dir, 'shuffled_train_filenames.npy'))

    if not os.path.exists(misc_dir):
        os.makedirs(misc_dir)

    print 'shuffled_train_filenames not found, generating ...'

    subfolders = [name for name in os.listdir(src_train_dir)
                  if os.path.isdir(os.path.join(src_train_dir, name))]

    train_filenames = []
    for subfolder in subfolders:
        train_filenames += glob.glob(src_train_dir + subfolder + '/*JPEG')

    train_filenames = np.asarray(sorted(train_filenames))

    np.random.seed(seed)
    np.random.shuffle(train_filenames)
    np.save(os.path.join(misc_dir, 'shuffled_train_filenames.npy'),
            train_filenames)

    return train_filenames


if __name__ == '__main__':
    with open('paths.yaml', 'r') as f:
        paths = yaml.load(f)

    train_img_dir = paths['train_img_dir']
    val_img_dir = paths['val_img_dir']
    misc_dir = paths['misc_dir']

    if len(sys.argv) < 2:
        gen_type = 'full'
    else:
        gen_type = sys.argv[1]

    if gen_type == 'full':
        print 'generating full dataset ...'
    elif gen_type == 'toy':
        print 'generating toy dataset ...'
    else:
        NotImplementedError("gen_type (2nd argument of make_hkl.py) can only be full or toy")

    train_filenames = get_train_filenames(train_img_dir, misc_dir)
    val_filenames = np.asarray(sorted(glob.glob(val_img_dir + '/*JPEG')))

    img_size = 256
    batch_size = 256

    if gen_type == 'toy':
        # generate 10 batches each
        train_filenames = train_filenames[:2560]
        val_filenames = val_filenames[:2560]

    for num_sub_batch in [1, 2]:
        tar_train_dir = paths['tar_train_dir']
        tar_val_dir = paths['tar_val_dir']
        tar_train_dir += '_b' + str(batch_size) + \
            '_b_' + str(batch_size / num_sub_batch)
        tar_val_dir += '_b' + str(batch_size) + \
            '_b_' + str(batch_size / num_sub_batch)

        # training data
        img_mean = save_batches(train_filenames, tar_train_dir,
                                img_size=img_size, batch_size=batch_size,
                                flag_avg=True, num_sub_batch=num_sub_batch)
        np.save(os.path.join(misc_dir, 'img_mean.npy'), img_mean)

        # validation data
        save_batches(val_filenames, tar_val_dir,
                     img_size=img_size, batch_size=batch_size,
                     num_sub_batch=num_sub_batch)
