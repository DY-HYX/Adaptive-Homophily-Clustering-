import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift, random_walker
from sklearn import preprocessing
import cv2
import math
from sklearn.decomposition import PCA


def LSC_superpixel(I, nseg):
    superpixelNum = nseg
    ratio = 0.075
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))
    superpixelLSC = cv2.ximgproc.createSuperpixelLSC(
        I,
        region_size=size,
        ratio=0.005)
    superpixelLSC.iterate()
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()
    return np.array(segments, np.int64)


def SEEDS_superpixel(I, nseg):
    I = np.array(I[:, :, 0:3], np.float32).copy()
    I_new = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    # I_new =np.array( I[:,:,0:3],np.float32).copy()
    height, width, channels = I_new.shape

    superpixelNum = nseg
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, int(superpixelNum), num_levels=2, prior=1,
                                               histogram_bins=5)
    seeds.iterate(I_new, 4)
    segments = seeds.getLabels()
    # segments=SegmentsLabelProcess(segments) # 排除labels中不连续的情况
    return segments


def SegmentsLabelProcess(labels):
    '''
    对labels做后处理，防止出现label不连续现象
    '''
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels

def color_results(arr2d, palette):
    arr_3d = np.zeros((arr2d.shape[0], arr2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr2d == c
        arr_3d[m] = i
    return arr_3d


class SLIC(object):
    def __init__(self, HSI, labels,n_segments=1000, compactness=20, max_iter=20, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        # 数据standardization标准化,即提前全局BN
        height, width, bands = HSI.shape  # 原始高光谱数据的三个维度
        data = np.reshape(HSI, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.labels =labels


    def get_Q_and_S_and_Segments(self):
        # 执行 SLCI 并得到Q(nxm),S(m*b)
        img = self.data
        (h, w, d) = img.shape  # 145*145*15
        # 计算超像素S以及相关系数矩阵Q  用法：https://vimsky.com/examples/usage/python-skimage.segmentation.slic-si.html
        segments = slic(img, n_segments=self.n_segments, compactness=self.compactness, max_num_iter=self.max_iter,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False)
        #########经过上述操作得到145*145*1的矩阵 矩阵值代表超像素0-195#######
        # segments = felzenszwalb(img, scale=1,sigma=0.5,min_size=25)

        # segments = quickshift(img,ratio=1,kernel_size=5,max_dist=4,sigma=0.8, convert2lab=False)

        # segments=LSC_superpixel(I=img, nseg= self.n_segments)

        # segments=SEEDS_superpixel(img,self.n_segments)

        # segments = sio.loadmat('labelsgt.mat')['labels']

        # segments = sio.loadmat('labels_hu_10000.mat')['labels']

        # 判断超像素label是否连续,否则予以校正
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))): segments = SegmentsLabelProcess(
            segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)



        # #######################显示超像素图片#########################
        out = mark_boundaries(img[:, :, [0, 1, 2]], segments)
        plt.imshow(out)
        plt.show()


        palette = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0),
                   5: (0, 255, 255), 6: (255, 0, 255), 7: (192, 192, 192), 8: (128, 128, 128), 9: (128, 0, 0),
                   10: (128, 128, 0), 11: (0, 128, 0), 12: (128, 0, 128), 13: (0, 128, 128), 14: (0, 0, 128),
                   15: (255, 165, 0), 16: (255, 215, 0)}
        gt = self.labels
        colored_gt = color_results(gt, palette)
        out2 = mark_boundaries(colored_gt, segments, color=(1, 0, 0))
        # plt.imshow(out2)
        # plt.show()


        # #######################显示超像素图片#########################
        segments = np.reshape(segments, [-1])  # 21025
        S = np.zeros([superpixel_count, d], dtype=np.float32)  # 196*15
        Q = np.zeros([w * h, superpixel_count], dtype=np.float32)  # 21025*196
        x = np.reshape(img, [-1, d])  # 21025*15

        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]  # 第一个是121*15
            superpixel = np.sum(pixels, 0) / count  # 求每个通道的均值 15*1  最终得到196个超像素的15个通道均值
            S[i] = superpixel
            Q[idx, i] = 1  # 标记了每个超像素的蒙板 21025*196

        self.S = S  # 得到超像素中的像素在每个通道的均值 196*15 每一行代表一个超像素的通道均值
        self.Q = Q  # 得到了每个超像素的蒙板每一列代表一个超像素蒙板

        return Q, S, self.segments

    def get_A(self, sigma: float):
        '''
         根据 segments 判定邻接矩阵
        :return:
        '''
        Edge_index = []
        Edge_atter = []
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)  # 196*196
        A_ones = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)  # 196*196
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                # if len(sub_set)>1:
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue

                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A[idx1, idx2] = A[idx2, idx1] = diss
                    A_ones[idx1, idx2] = A_ones[idx2, idx1] = 1
                    a = [sub_min, sub_max]
                    b = [sub_max, sub_min]
                    if a not in Edge_index:
                        Edge_index.append(a)
                        Edge_index.append(b)
                        Edge_atter.append(diss)
                        Edge_atter.append(diss)
        Edge_index2 = np.array(Edge_index)
        Edge_index2 = Edge_index2.transpose(1, 0)
        Edge_atter2 = np.array(Edge_atter)
        return A, Edge_index2.astype('int64'), Edge_atter2.astype('int64') ,A_ones


class LDA_SLIC(object):
    def __init__(self, data,labels, n_component):
        self.data = data  # 原始图像145*145*200
        # self.init_labels = labels  # 原始标签145*145*1
        self.curr_data = data
        self.n_component = n_component  # 15
        self.height, self.width, self.bands = data.shape
        self.x_flatt = np.reshape(data, [self.width * self.height, self.bands])  # 图像由145*145*200转为20125*200
        # self.y_flatt = np.reshape(labels, [self.height * self.width])
        self.labes = labels  # 训练标签 145*145*1

    # def LDA_Process(self, curr_labels):
    #     '''
    #     :param curr_labels: height * width
    #     :return:
    #     '''
    #     curr_labels = np.reshape(curr_labels, [-1])  # 训练标签 21025*1
    #     idx = np.where(curr_labels != 0)[0]  # 标签非零处索引
    #     x = self.x_flatt[idx]  # 原始图片中训练像素 1031*200
    #     y = curr_labels[idx]  # 非零标签  1031
    #     lda = LinearDiscriminantAnalysis()  # n_components=self.n_component  https://zhuanlan.zhihu.com/p/137968371
    #     lda.fit(x, y - 1)  # y-1代表所有标签值减去1
    #     X_new = lda.transform(
    #         self.x_flatt)  # 21025*15 这里的15是计算出来的，跟class_count-1无关    https://scikit-learn.org.cn/view/618.html
    #     return np.reshape(X_new, [self.height, self.width, -1])

    def applyPCA(self, X, numComponents):

        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

        return newX

    def SLIC_Process(self, img, scale=25):
        n_segments_init = int((self.height * self.width) / scale)  # 片段  分为145*145/100的片段  210
        print("n_segments_init", n_segments_init)
        myslic = SLIC(img, n_segments=n_segments_init, labels=self.labes,   compactness=0.006, sigma=1,
                      min_size_factor=0.1, max_size_factor=2)  # ip0.06 SA=0.005 PU 0.006 Tr 0.01 HU 0.0005
        Q, S, Segments = myslic.get_Q_and_S_and_Segments()
        A, Edge_index, Edge_atter,A_ones = myslic.get_A(sigma=10)
        return Q, S, A, Edge_index, Edge_atter, Segments,A_ones

        ####### S(196*15)得到超像素中的像素在每个通道的均值 196*15 每一行代表一个超像素的通道均值###
        ################ Q(20125*196)得到了每个超像素的蒙板每一列代表一个超像素蒙板#############
        ############segments 145*145*1 原始图片超像素标签#################################
        ########################A 196*196 超像素之间的距离###############################

    def simple_superpixel(self, scale):
        # curr_labels = self.init_labels  # 原始标签145*145*1
        # X = self.LDA_Process(curr_labels)  # 降维后的数据21025*15
        # X = self.data
        X = self.applyPCA(self.data, 10)#ip 60 SA 70  PU 10 TR 20  HU 10
        Q, S, A, Edge_index, Edge_atter, Seg,A_ones = self.SLIC_Process(X, scale=scale)
        return Q, S, A, Edge_index, Edge_atter, Seg,A_ones

