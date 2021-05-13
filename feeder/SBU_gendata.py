import numpy as np
import argparse
import os
import sys
from SBU_read_skeleton import read_xyz
from numpy.lib.format import open_memmap
import pickle

FOLDS={"FOLD_1": ("s01s02", "s03s04", "s05s02", "s06s04"),
       "FOLD_2": ("s02s03", "s02s07", "s03s05", "s05s03"),
       "FOLD_3": ("s01s03", "s01s07", "s07s01", "s07s03"),
       "FOLD_4":("s02s01", "s02s06", "s03s02", "s03s06"),
       "FOLD_5":("s04s02", "s04s03", "s04s06", "s06s02", "s06s03")}

max_body = 2
num_joint = 15
max_frame = 55
toolbar_width = 30


def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            fold_num='FOLD_1',
            part='eval'):
    
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if(filename in FOLDS[fold_num]):
            istraining=False
        else:
            istraining=True
        if(filename in [".DS_Store"]):
            continue

        for action_directory in os.listdir(os.path.join(data_path,filename)):
            if(action_directory in [".DS_Store"]):
                continue
            action_class = int(action_directory)
            for sequence in os.listdir(os.path.join(data_path,filename,action_directory)): 
                if(sequence in [".DS_Store"]):
                    continue

                if part == 'train':
                    issample = istraining
                elif part == 'val':
                    issample = not (istraining)
                else:
                    raise ValueError()

                if issample:
                    sample_name.append(os.path.join(data_path,filename,action_directory,sequence,"skeleton_pos.txt"))
                    sample_label.append(action_class - 1)
        
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
    # https://blog.csdn.net/u014630431/article/details/72844501
    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    fl = open_memmap(
        '{}/{}_num_frame.npy'.format(out_path, part),
        dtype='int',
        mode='w+',
        shape=(len(sample_label),))

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), fold_num, part))
        data = read_xyz(s, max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
        

        fl[i] = data.shape[1] # num_frame
    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SBU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='../SBU-Kinect-Interaction')
    parser.add_argument('--out_folder', default='../data0/SBU-RGB-D')

    fold_nums = ['FOLD_1', 'FOLD_2','FOLD_3', 'FOLD_4', 'FOLD_5']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in fold_nums:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                fold_num=b,
                part=p)
