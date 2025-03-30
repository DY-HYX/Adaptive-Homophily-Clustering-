from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import scipy.io as sio
import math
from sklearn import preprocessing
from munkres import Munkres
from sklearn import metrics
import torch
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import os
import argparse
from ruamel.yaml import YAML
from termcolor import cprint
from torch_geometric.data import Data
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#选择cpu或者GPU

def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 排序取索引
def choose_top(image,cornor_index,x,y,patch,b,n_top):
    sort = image.reshape(patch * patch, b)
    sort = torch.from_numpy(sort).type(torch.FloatTensor)
    pos = (x - cornor_index[0]) * patch + (y - cornor_index[1])
    Q = torch.sum(torch.pow(sort[pos] - sort, 2), dim=1)
    _, indices = Q.topk(k=n_top, dim=0, largest=False, sorted=True)
    return indices
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(pca_image, point, i, patch, W, H):
    x = point[i,0]
    y = point[i,1]
    m=int((patch-1)/2)##patch奇数
    _,_,b=pca_image.shape
    if x<=m:
        if y<=m:
            temp_image = pca_image[0:patch, 0:patch, :]
            cornor_index = [0,0]
        if y>=(H-m):
            temp_image = pca_image[0:patch, H-patch:H, :]
            cornor_index = [0, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[0:patch, y-m:y+m+1, :]
            cornor_index = [0, y-m]
    if x>=(W-m):
        if y<=m:
            temp_image = pca_image[W-patch:W, 0:patch, :]
            cornor_index = [W-patch, 0]
        if y>=(H-m):
            temp_image = pca_image[W-patch:W, H-patch:H, :]
            cornor_index = [W - patch, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[W-patch:W, y-m:y+m+1, :]
            cornor_index = [W - patch, y-m]
    if x>m and x<W-m:
        if y<=m:
            temp_image = pca_image[x-m:x+m+1, 0:patch, :]
            cornor_index = [x-m, 0]
        if y>=(H-m):
            temp_image = pca_image[x-m:x+m+1, H-patch:H, :]
            cornor_index = [x - m, H-patch]
        if y>m and y<H-m:
            temp_image = pca_image[x-m:x+m+1, y-m:y+m+1, :]
            cornor_index = [x - m, y-m]
            # look11=pca_image[:,:,0]
            # look12=temp_image[:,:,0]
            # print(temp_image.shape)
    center_pos = (x - cornor_index[0]) * patch + (y - cornor_index[1])
    return temp_image,cornor_index,center_pos
# 汇总训练数据和测试数据
def train_and_test_data(pca_image, band, train_point, test_point, true_point, patch, w, h):
    x_train = torch.zeros((train_point.shape[0], patch, patch, band), dtype=torch.float32).cuda()
    x_test = torch.zeros((test_point.shape[0], patch, patch, band), dtype=torch.float32).cuda()
    x_true = torch.zeros((true_point.shape[0], patch, patch, band), dtype=torch.float32).cuda()
    corner_train = np.zeros((train_point.shape[0], 2), dtype=int)
    corner_test = np.zeros((test_point.shape[0], 2), dtype=int)
    corner_true = np.zeros((true_point.shape[0], 2), dtype=int)
    center_pos_train = torch.zeros((train_point.shape[0]), dtype=int).cuda()
    center_pos_test = torch.zeros((test_point.shape[0]), dtype=int).cuda()
    center_pos_ture = torch.zeros((true_point.shape[0]), dtype=int).cuda()
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:],corner_train[i,:],center_pos_train[i]= gain_neighborhood_pixel(pca_image, train_point, i, patch, w, h)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:],corner_test[j,:],center_pos_test[j] = gain_neighborhood_pixel(pca_image, test_point, j, patch, w, h)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:],corner_true[k,:],center_pos_ture[k] = gain_neighborhood_pixel(pca_image, true_point, k, patch, w, h)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")

    return x_train, x_test, x_true,corner_train,corner_test,corner_true,center_pos_train,center_pos_test,center_pos_ture
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(net,input_normalize, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_target,y_train_flatten) in enumerate(train_loader):

        batch_target = batch_target.cuda()
        optimizer.zero_grad()
        batch_pred = net(input_normalize,y_train_flatten)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = y_train_flatten.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def test_epoch(net,input_normalize, test_loader, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_target,y_test_flatten) in enumerate(test_loader):

        batch_target = batch_target.cuda()
        batch_pred = net(input_normalize,y_test_flatten)
        loss = criterion(batch_pred, batch_target)
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = y_test_flatten.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


def GET_A(temp_image,D,corner,l,sigma=10, w_all=145, h_all=145):#l为邻域范围，sigma为计算距离的参数
    N,h,w,_=temp_image.shape
    B = np.zeros((w * h, w * h), dtype=np.float32)
    for i in range(h):  # 图像的行  h代表有几行，w代表有几列
        for j in range(w):  # 图像的列
            m = int(i * w + j)  # 在邻接矩阵中的行数
            for k in range(l):  # 邻域的行数
                for q in range(l):  # 邻域的列数
                    n = int((i + (k - (l - 1) / 2)) * w + (j + (q - (l - 1) / 2)))  # 计算邻域，并转换为邻域在邻接矩阵中的列数
                    if 0 <= i + (k - (l - 1) / 2) < h and 0 <= (j + (q - (l - 1) / 2)) < w and m != n:
                        B[m, n] = 1
##############得到每个输入图像的7*7图像的邻域矩阵###############################
    index=np.argwhere(B == 1)#得到邻域矩阵中不为零的索引
    index_num,_=index.shape
    X = np.zeros((index_num,2),dtype=np.int64)
    Y = np.zeros((index_num, 2), dtype=np.int64)
    for i in range(index_num):
        X[i, 0] = index[i, 0] // w #邻域矩阵行值在图像中行坐标
        X[i, 1] = index[i, 0] % w#邻域矩阵行值在图像中列坐标
        Y[i, 0] = index[i, 1] // w#邻域矩阵列值在图像中行坐标
        Y[i, 1] = index[i, 1] % w#邻域矩阵列值在图像中列坐标
##############得到每个输入图像的7*7图像的二维坐标###############################
    X_N = np.zeros((N,index_num,2), dtype=np.int64)#在原始图像上的行值
    Y_N = np.zeros((N, index_num, 2), dtype=np.int64)#在原始图像上的列值
    corner_N = np.expand_dims(corner, 1).repeat(index_num, axis=1)
    X=np.expand_dims(X, 0).repeat(N, axis=0)
    Y = np.expand_dims(Y, 0).repeat(N, axis=0)
    X_N[:, :, 0] = X[:,:,0] + corner_N[:,:,0]#在原始图像上的行值
    X_N[:, :, 1] = X[:,:,1] + corner_N[:,:,1]#在原始图像上的列值
    X_A=X_N[:, :, 0]*w_all+X_N[:, :, 1]#在原始图像邻接矩阵距离索引
    Y_N[:, :, 0] = Y[:,:,0] + corner_N[:,:,0]#在原始图像上的行值
    Y_N[:, :, 1] = Y[:,:,1] + corner_N[:,:,1]#在原始图像上的列值
    Y_A = Y_N[:, :, 0] * w_all + Y_N[:, :, 1]#在原始图像邻接矩阵距离索引

    A = np.zeros((N, w * h, w * h), dtype=np.float32)
    A = torch.from_numpy(A).type(torch.FloatTensor).cuda()
    index2 = np.where(B == 1)  # 得到邻域矩阵中不为零的索引
    for i in range(N):
        C=torch.from_numpy(B).type(torch.FloatTensor).cuda()
        C[index2[0],index2[1]]= D[X_A[i], Y_A[i]]
        A[i,:,:] = C

    return A

def GET_A2(temp_image,input2,corner,patches,l,sigma=10,):#l为邻域范围，sigma为计算距离的参数
    input2=input2.cuda()
    N,h,w,_=temp_image.shape
    B = np.zeros((w * h, w * h), dtype=np.float32)
    for i in range(h):  # 图像的行  h代表有几行，w代表有几列
        for j in range(w):  # 图像的列
            m = int(i * w + j)  # 在邻接矩阵中的行数
            for k in range(l):  # 邻域的行数
                for q in range(l):  # 邻域的列数
                    n = int((i + (k - (l - 1) / 2)) * w + (j + (q - (l - 1) / 2)))  # 计算邻域，并转换为邻域在邻接矩阵中的列数
                    if 0 <= i + (k - (l - 1) / 2) < h and 0 <= (j + (q - (l - 1) / 2)) < w and m != n:
                        B[m, n] = 1
##############得到每个输入图像的7*7图像的邻域矩阵###############################
    index=np.argwhere(B == 1)#得到邻域矩阵中不为零的索引
    index2 = np.where(B == 1)  # 得到邻域矩阵中不为零的索引
    A = np.zeros((N, w * h, w * h), dtype=np.float32)

    for i in range(N):#####corenor为左上角的值
        C = np.array(B)
        x_l=int(corner[i,0])
        x_r=int(corner[i,0]+patches)
        y_l=int(corner[i,1])
        y_r=int(corner[i,1]+patches)
        D = pdists_corner(input2[x_l:x_r,y_l:y_r,:],sigma)
        D = D.cpu().numpy()
        m= D[index2[0],index2[1]]
        C[index2[0], index2[1]] = D[index2[0],index2[1]]
        A[i,:,:] = C
    A = torch.from_numpy(A).type(torch.FloatTensor).cuda()
    return A


def pdists_corner(A,sigma=10):
    height,width, band = A.shape
    A=A.reshape(height * width, band)
    prod = torch.mm(A, A.t())#21025*21025
    norm = prod.diag().unsqueeze(1).expand_as(prod)#21025*21025
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    D =torch.exp(-res/(sigma ** 2))
    return D

def pdists(A,sigma=10):
    A=A.cuda()
    prod = torch.mm(A, A.t())#21025*21025
    norm = prod.diag().unsqueeze(1).expand_as(prod)#21025*21025
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    D =torch.exp(-res/(sigma ** 2))
    return D

def normalize(data):
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)#计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布
    data = np.reshape(data, [height, width, bands])
    return data



################get data######################################################################################################################
def load_dataset(Dataset):
    if Dataset == 'Indian':
        # mat_data = sio.loadmat('/home/yat/datasets/Indian_pines_corrected.mat')
        # mat_gt = sio.loadmat('/home/yat/datasets/Indian_pines_gt.mat')
        mat_data = sio.loadmat('/home/tsing/data/datasets/Indian_pines_corrected.mat')
        mat_gt = sio.loadmat('/home/tsing/data/datasets/Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'PaviaU':
        uPavia = sio.loadmat('/home/tsing/data/datasets/PaviaU.mat')
        gt_uPavia = sio.loadmat('/home/tsing/data/datasets/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'Pavia':
        uPavia = sio.loadmat('/home/tsing/data/datasets/Pavia.mat')
        gt_uPavia = sio.loadmat('/home/tsing/data/datasets/Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        TOTAL_SIZE = 148152
        VALIDATION_SPLIT = 0.999
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'Salinas':
        SV = sio.loadmat('/home/tsing/data/datasets/Salinas_corrected.mat')
        gt_SV = sio.loadmat('/home/tsing/data/datasets/Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'KSC':
        KSC = sio.loadmat('/home/tsing/data/datasets/KSC.mat')
        gt_KSC = sio.loadmat('/home/tsing/data/datasets/KSC_gt.mat')
        data_hsi = KSC['KSC']
        gt_hsi = gt_KSC['KSC_gt']
        TOTAL_SIZE = 5211
        VALIDATION_SPLIT = 0.95
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'Botswana':
        BS = sio.loadmat('/home/tsing/data/datasets/Botswana.mat')
        gt_BS = sio.loadmat('/home/tsing/data/datasets/Botswana_gt.mat')
        data_hsi = BS['Botswana']
        gt_hsi = gt_BS['Botswana_gt']
        TOTAL_SIZE = 3248
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'Botswana':
        BS = sio.loadmat('/home/tsing/data/datasets/Botswana.mat')
        gt_BS = sio.loadmat('/home/tsing/data/datasets/Botswana_gt.mat')
        data_hsi = BS['Botswana']
        gt_hsi = gt_BS['Botswana_gt']
        TOTAL_SIZE = 3248
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'Houston':
        BS = sio.loadmat('/home/tsing/data/datasets/Houston.mat')
        gt_BS = sio.loadmat('/home/tsing/data/datasets/Houston_gt.mat')
        data_hsi = BS['Houston']
        gt_hsi = gt_BS['Houston_gt']
        TOTAL_SIZE = 664845
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)


    if Dataset == 'Trento':
        Tr = sio.loadmat('/home/tsing/data/datasets/Trento.mat')
        gt_Tr = sio.loadmat('/home/tsing/data/datasets/Trento_gt.mat')
        data_hsi = Tr['HSI']
        gt_hsi= gt_Tr['gt']
        gt_hsi[np.where(gt_hsi ==-1)]=0
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT


def sampling(proportion, ground_truth,CLASSES_NUM):
    train = {}
    test = {}
    train_num = []
    test_num = []
    labels_loc = {}
    for i in range(CLASSES_NUM):
        indexes = np.argwhere(ground_truth == (i + 1))
        np.random.shuffle(indexes)#打乱顺序
        labels_loc[i] = indexes
        if proportion != 1:
            # nb_val = max(int((1 - proportion) * len(indexes)), 3)
            if indexes.shape[0]<=60:
                nb_val = 15
            else:
                nb_val = 30
        else:
            nb_val = 0
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        train_num.append(nb_val)
        test_num.append(len(indexes)-nb_val)
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes =train[0]
    test_indexes = test[0]
    for i in range(CLASSES_NUM-1):
        train_indexes= np.concatenate((train_indexes,train[i+1]),axis=0)
        test_indexes= np.concatenate((test_indexes,test[i+1]),axis=0)
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes,train_num,test_num#返回训练集和测试集的索引


def index_change(index,w):
    N=len(index)
    index2=np.zeros((N,2),dtype=int)
    for i in range(N):
        index2[i, 0] = index[i] // w
        index2[i, 1] = index[i] % w
    return index2
def get_label(indices,gt_hsi):
    dim_0 = indices[:, 0]
    dim_1 = indices[:, 1]
    label=gt_hsi[dim_0,dim_1]
    return label

def get_label_flatten(indices,width):
    dim_0 = indices[:, 0]
    dim_1 = indices[:, 1]
    label=dim_0*width+dim_1
    return label

def get_data(dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择cpu或者GPU
    data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(dataset)
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
    CLASSES_NUM = max(gt)
    train_indices, test_indices,train_num,test_num = sampling(VALIDATION_SPLIT, gt_hsi, CLASSES_NUM)
    _, total_indices,_,total_num = sampling(1, gt_hsi , CLASSES_NUM)
    y_train = get_label(train_indices, gt_hsi)-1
    y_test = get_label(test_indices, gt_hsi)-1
    y_true = get_label(total_indices, gt_hsi)-1
    height, width = gt_hsi.shape
    y_train_flatten=get_label_flatten(train_indices,width)
    y_test_flatten = get_label_flatten(test_indices, width)
    y_train_flatten = torch.from_numpy(y_train_flatten).to(device)
    y_test_flatten = torch.from_numpy(y_test_flatten).to(device)
    return  data_hsi,CLASSES_NUM, y_true, gt, gt_hsi

# def metrics(best_OA2, best_AA_mean2, best_Kappa2,AA2):
#     results = {}
#     results["OA"] = best_OA2 * 100.0
#     results['AA'] = best_AA_mean2 * 100.0
#     results["Kappa"] = best_Kappa2 * 100.0
#     results["class acc"] = AA2 * 100.0
#     return results
def show_results(results, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["OA"] for r in results]
        aa = [r['AA'] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]

        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)

    else:
        accuracy = results["OA"]
        aa = results['AA']
        classacc = results["class acc"]
        kappa = results["Kappa"]

    text += "---\n"
    text += "class acc :\n"
    if agregated:
        for score, std in zip(class_acc_mean,
                                     class_acc_std):
            text += "\t{:.02f} +- {:.02f}\n".format(score, std)
    else:
        for score in classacc:
            text += "\t {:.02f}\n".format(score)
    text += "---\n"

    if agregated:
        text += ("OA: {:.02f} +- {:.02f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
        text += ("AA: {:.02f} +- {:.02f}\n".format(np.mean(aa),
                                                   np.std(aa)))
        text += ("Kappa: {:.02f} +- {:.02f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "OA : {:.02f}%\n".format(accuracy)
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)

    print(text)


def data_process(S, Edge_index, Edge_atter,y,num_node):
    data = Data(x=S, edge_index=Edge_index, edge_attr=Edge_atter, y=y)
    data.num_nodes = num_node
    data.num_edges = Edge_index.shape[1]
    data.batch = torch.zeros(num_node, dtype=torch.int64).to(device)
    # data = graphormer_pre_processing(
    #     data,
    #     20
    # )
    return data

def spixel_to_pixel_labels(sp_level_label, association_mat):
    sp_level_label = np.reshape(sp_level_label, (-1, 1))
    pixel_level_label = np.matmul(association_mat, sp_level_label).reshape(-1)
    return pixel_level_label.astype('int')

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def class_acc(y_true, y_pre):
    """
    calculate each class's acc
    :param y_true:
    :param y_pre:
    :return:
    """
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
        y_c_p = y_pre[np.nonzero(y_true == c)]
        acurracy = metrics.accuracy_score(y_c, y_c_p)
        ca.append(acurracy)
    ca = np.array(ca)
    return ca

def cluster_accuracy(y_true, y_pre, return_aligned=False):
    y_true = y_true.astype('float32')
    y_pre = y_pre.astype('float32')
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]

    # # calculate accuracy
    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    acc = 1. - missrate
    nmi = metrics.normalized_mutual_info_score(y_true, y_pre)
    kappa = metrics.cohen_kappa_score(y_true, y_best)
    ca = class_acc(y_true, y_best)
    ari = metrics.adjusted_rand_score(y_true, y_best)
    fscore = metrics.f1_score(y_true, y_best, average='micro')
    pur = purity_score(y_true, y_best)
    if return_aligned:
        return y_best, acc, kappa, nmi, ari, pur, ca
    return acc, kappa, nmi, ari, pur, ca


def get_args_key(args):
    return "-".join([args.model_name])

def get_args(model_name, yaml_path=None) -> argparse.Namespace:
    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    parser = argparse.ArgumentParser(description='Parser for Simple Unsupervised Graph Representation Learning')
    # Basics
    parser.add_argument("--num-gpus-total", default=1, type=int)
    parser.add_argument("--num-gpus-to-use", default=1, type=int)
    parser.add_argument("--black-list", default=None, type=int, nargs="+")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--model_name", default=model_name)
    parser.add_argument("--save_model", default=False)
    parser.add_argument("--seed", default=0)
    # Dataset
    parser.add_argument('--data-root', default="~/graph-data", metavar='DIR', help='path to dataset')
    # Pretrain
    parser.add_argument("--pretrain", default=False, type=bool)
    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr2', '--learning-rate2', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate2', dest='lr2')
    parser.add_argument("--use-bn", default=False, type=bool)
    parser.add_argument("--perf-task-for-val", default="Node", type=str)  # Node or Link
    parser.add_argument('--w_loss1', type=float, default=1, help='')
    parser.add_argument('--w_loss2', type=float, default=1, help='')
    parser.add_argument('--w_loss3', type=float, default=1, help='')
    parser.add_argument('--margin1', type=float, default=0.8, help='')
    parser.add_argument('--margin2', type=float, default=0.2, help='')
    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")
    # Update params from .yamls
    args = parser.parse_args()
    return args

def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT: {}".format(get_args_key(_args)), "yellow")
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))




def get_dataset(dataset):
    data = dataset
    # data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    i = torch.LongTensor([data.edge_index[0].numpy(), data.edge_index[1].numpy()])
    v = torch.FloatTensor(torch.ones([data.num_edges]))
    A_sp = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))


    A = A_sp.to_dense()
    I = torch.eye(A.shape[1]).to(A.device)
    A_I = A + I
    # A_nomal = normalize_graph(A)
    A_I_nomal = normalize_graph(A_I)#归一化邻接矩阵
    A_I_nomal = A_I_nomal.to_sparse()

    lable = data.y
    nb_feature = data.num_features
    nb_classes = int(lable.max() - lable.min()) + 1
    nb_nodes = data.num_nodes
    data.x = torch.FloatTensor(data.x)
    eps = 2.2204e-16
    norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
    data.x = data.x.div(norm.expand_as(data.x))
    adj_1 = csr_matrix(
        (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
        shape=(data.num_nodes, data.num_nodes))#邻接关系索引表达形式

    return data, [A_I_nomal,adj_1], [data.x], [lable, nb_feature, nb_classes, nb_nodes]

# def cut(X,n):
#     for i in range(0:n:X.shape[])
def set_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
