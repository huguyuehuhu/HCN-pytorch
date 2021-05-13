# sys
import pickle

# torch
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import random

try:
    from utils import utils
    from feeder import tools
except:
    import tools, utils

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 num_frame_path,
                 random_valid_choose=False,
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 window_size=-1,
                 normalization=False,
                 debug=False,
                 origin_transfer=False,
                 p_interval=1,
                 crop_resize=False,
                 rand_rotate=0,
                 mmap=False,
                 ):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path = num_frame_path
        self.random_choose = random_choose
        self.random_valid_choose = random_valid_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.origin_transfer = origin_transfer
        self.p_interval = p_interval
        self.crop_resize = crop_resize
        self.rand_rotate = rand_rotate
        self.mmap = mmap
        self.load_data()
        
        self.indices=list(range(len(self.data))) 
        self.coordinate_transfer()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        # load label
        if '.pkl' in self.label_path:
            try:
                with open(self.label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(
                        f, encoding='latin1')
        # old label format
        elif '.npy' in self.label_path:
            self.label = list(np.load(self.label_path))
            self.sample_name = [str(i) for i in range(len(self.label))]
        else:
            raise ValueError()

        # load data
        if self.mmap == True:
            self.data = np.load(self.data_path,mmap_mode='r')
        else:
            self.data = np.load(self.data_path,mmap_mode=None) # directly load all data in memory, it more efficient but memory resource consuming for big file

        # load num of valid frame length
        self.valid_frame_num = np.load(self.num_frame_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.num_frame_path = self.num_frame_path[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def coordinate_transfer(self):
        data_numpy = self.data

        if self.origin_transfer == 2:
            #  take joints 2  of each person, as origins of each person
            origin = data_numpy[:, :, :, 1, :]
            data_numpy = data_numpy - origin[:, :, :, None, :]
        elif self.origin_transfer == 0:
            #  take joints 2  of first person, as origins of each person
            origin = data_numpy[:, :, :, 1, 0]
            data_numpy = data_numpy - origin[:, :, :, None, None]
            

        elif self.origin_transfer == 1:
            #  take joints 2  of second person, as origins of each person
            origin = data_numpy[:, :, :, 1, 1]
            data_numpy = data_numpy - origin[:, :, :, None, None]
        else:
            # print('no origin transfer')
            pass

        self.data = data_numpy

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(
            axis=2, keepdims=True).mean(
            axis=4, keepdims=True).mean(axis=0)
        # mean_map 3,1,25,1
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

        self.min_map = (data.transpose(0, 2, 3, 4, 1).reshape(N * T * V * M, C)).min(axis=0)
        self.max_map = (data.transpose(0, 2, 3, 4, 1).reshape(N * T * V * M, C)).max(axis=0)
        self.mean_mean = (data.transpose(0, 2, 3, 4, 1).reshape(N * T * V * M, C)).mean(axis=0)
        print('min_map', self.min_map, 'max_map', self.max_map, 'mean', self.mean_mean)

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        # input: C, T, V, M
        """if(index==0):
            random.shuffle(self.indices)
        
        index=self.indices[index]
        """
        data_numpy = self.data[index]
        # if self.mmap = True, the loaded data_numpy is read-only, and torch.utils.data.DataLoader could load type 'numpy.core.memmap.memmap'
        if self.mmap:
            data_numpy = np.array(data_numpy) # convert numpy.core.memmap.memmap to numpy

        label = self.label[index]
        valid_frame_num = self.valid_frame_num[index]

        # normalization
        if self.normalization:
            # data_numpy = (data_numpy - self.mean_map) / self.std_map
            # be careful the value is for NTU_RGB-D, for other dataset, please replace with value from function 'get_mean_map'
            if self.origin_transfer == 0:
                min_map, max_map = np.array([-4.9881773, -2.939787, -4.728529]), np.array(
                    [5.826573, 2.391671, 4.824233])
            elif self.origin_transfer == 1:
                min_map, max_map = np.array([-5.836631, -2.793758, -4.574943]), np.array([5.2021008, 2.362596, 5.1607])
            elif self.origin_transfer == 2:
                min_map, max_map = np.array([-2.965678, -1.8587272, -4.574943]), np.array(
                    [2.908885, 2.0095677, 4.843938])
            else:
                pass
            
            C, T, V, M = data_numpy.shape
            max_map=data_numpy[0:valid_frame_num,:,:,:].transpose(1, 2, 3, 0).reshape( T * V * M, C).max(axis=0)
            min_map=data_numpy[0:valid_frame_num,:,:,:].transpose(1, 2, 3, 0).reshape( T * V * M, C).min(axis=0)
        
            #data_numpy = (255.0*(data_numpy - min_map[:, None, None, None])) / (255.0*(max_map[:, None, None, None] - min_map[:, None, None, None]))
            data_numpy = (data_numpy - min_map[:, None, None, None]) / (max_map[:, None, None, None] - min_map[:, None, None, None])
            
        # processing
        
        if self.crop_resize:
            data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
            

        if self.rand_rotate > 0:
            data_numpy = tools.rand_rotate(data_numpy, self.rand_rotate)

        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size, auto_pad=True)


        if self.random_valid_choose:
            data_numpy = tools.valid_choose(data_numpy, self.window_size, random_pad = True)
        elif self.window_size > 0 and (not self.crop_resize) and (not self.random_choose):
            data_numpy = tools.valid_choose(data_numpy, self.window_size, random_pad=False)


        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)

        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        
        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def crop(seff, data, tw, th):
        _, w, h, _ = data.shape
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return data[:, x1:x1 + tw, y1:y1 + th, :]


def test(data_path, label_path, vid=None):
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        pose, = ax.plot(np.zeros(V * M), np.zeros(V * M), 'g^')
        ax.axis([-1, 1, -1, 1])

        for n in range(N):
            for t in range(T):
                x = data[n, 0, t, :, 0]
                y = data[n, 1, t, :, 0]
                z = data[n, 2, t, :, 0]
                pose.set_xdata(x)
                pose.set_ydata(y)
                fig.canvas.draw()
                plt.pause(0.1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    data_path = "/data0/NTU-RGB-D/xview/val_data.npy"
    label_path = "/data0/NTU-RGB-D/xview/val_label.pkl"
    dataset = Feeder(data_path, label_path, random_valid_choose=False,
                     random_shift=False,
                     random_move=False,
                     window_size=100,
                     normalization=False,
                     debug=False,
                     origin_transfer=False,  # could not work here
                     fft=None)
    print(np.bincount(dataset.label))

    test(data_path, label_path, vid='S001C002P001R001A058')










