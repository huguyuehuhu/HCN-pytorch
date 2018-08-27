import numpy as np
import random
import torch.nn.functional as F
import torch

def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M),dtype=data_numpy.dtype)
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def valid_choose(data_numpy, window_size, random_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0  # not all person,joints,ordinate is zero(must used be for normalization, , as validate frame
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    valid_size = end - begin
    if valid_size > window_size:
        bias = random.randint(0, valid_size - window_size-1)
        data_choose = np.zeros((C, window_size, V, M), dtype=data_numpy.dtype)
        data_choose[:,0:window_size,:,:] = data_numpy[:, begin+bias:begin+bias+window_size, :, :]
        return data_choose
    else:
        data_choose = np.zeros((C, valid_size, V, M), dtype=data_numpy.dtype)
        data_choose[:,0:valid_size,:,:] = data_numpy[:, begin:end, :, :]
        return auto_pading(data_choose, window_size, random_pad=random_pad)

def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    ## valid1: by estimated from data
    # valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0  # not all person,joints,ordinate is zero(must used be for normalization, , as validate frame
    # begin = valid_frame.argmax()
    # end = len(valid_frame) - valid_frame[::-1].argmax()
    # valid_size = end - begin

    ## valid2: generated when generating dataset
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64),valid_size)# constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:,begin+bias:begin+bias+cropped_length, :, :]

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, :, :, None]
    data = F.upsample(data, size=(window, 1), mode='bilinear',align_corners=False).squeeze(dim=3).squeeze(dim=0) # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data


def rand_rotate(data_numpy,rand_rotate):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape

    R = np.eye(3)
    for i in range(3):
        theta = (np.random.rand()*2 -1)*rand_rotate * np.pi
        Ri = np.eye(3)
        Ri[C - 1, C - 1] = 1
        Ri[0, 0] = np.cos(theta)
        Ri[0, 1] = np.sin(theta)
        Ri[1, 0] = -np.sin(theta)
        Ri[1, 1] = np.cos(theta)
        R = R * Ri

    data_numpy = np.matmul(R,data_numpy.reshape(C,T*V*M)).reshape(C,T,V,M).astype('float32')
    return  data_numpy


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape,dtype=data_numpy.dtype)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0 # not all person,joints,ordinate is zero(must used be for normalization, , as validate frame
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift



