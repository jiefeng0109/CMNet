"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from processing_library import *
from cv2 import *
from PIL import Image
import scipy.io as sio
import math
from processing_library_cnn import *
from metrics_torch import sam, psnr, scc, ergas,mse
import random
# from cnn_torch import CNN

# os.environ["CUDA_VISIBLE_DEVICES"]='1'
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

batch_size = 8
step = 1
display_step = 50


index = batch_size
learning_rate_PNN = 0.00001
num_epoch_PNN = 1000
dim_input_class = 27
w=27

def show_hsi(fuse,flag):
    if flag == 1:#pavia center
        fuse[fuse<0]=0
        fuse[fuse>1]=1
        rgb_datas = fuse[:, :, (80, 40, 20)]
        # rgb_datas = fuse[:, :, (25, 15, 20)]
        bgr_datas = rgb_datas[:, :, (2, 1, 0)]
        bgr_datas -= np.min(bgr_datas)
        bgr_datas /= np.max(bgr_datas)
        bgr_datas *= 255
        return bgr_datas
    if flag == 2:  # Botswana
        fuse[fuse<0]=0
        fuse[fuse>1]=1
        rgb_datas = fuse[:, :, (80, 90, 100)]
        bgr_datas = rgb_datas[:, :, (2, 1, 0)]
        bgr_datas -= np.min(bgr_datas)
        bgr_datas /= np.max(bgr_datas)
        bgr_datas *= 255
        return bgr_datas
    if flag == 3:  # Chikusei
        fuse[fuse<0]=0
        fuse[fuse>1]=1
        rgb_datas = fuse[:, :, (80, 40, 20)]
        bgr_datas = rgb_datas[:, :, (2, 1, 0)]
        bgr_datas -= np.min(bgr_datas)
        bgr_datas /= np.max(bgr_datas)
        bgr_datas *= 255
        return bgr_datas

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        # conv   = self.bn(conv)
        relu = self.relu(conv)
        return relu


def load_data(data_name):
    '''读取数据'''
    path = os.getcwd()
    pre = sio.loadmat('...\chikusei_pre.mat')
    LRHS = pre['LRHS']
    PAN = pre['PAN']

    LRHS_train_patch = pre['LRHS_train_patch']
    PAN_train_patch = pre['PAN_train_patch']
    LRHS_train_gt = pre['LRHS_train_gt']

    LRHS_test_patch = pre['LRHS_test_patch']
    LRHS_test_gt = pre['LRHS_test_gt']
    PAN_test_patch = pre['PAN_test_patch']

    data_orig_norm = np.zeros([PAN.shape[0],PAN.shape[1],LRHS.shape[-1]])


    return data_orig_norm, LRHS, PAN, LRHS_train_patch, PAN_train_patch, LRHS_train_gt, LRHS_test_patch, LRHS_test_gt, PAN_test_patch


##############################################################################
data_orig_norm, LRHS, PAN, LRHS_train_patch, PAN_train_patch, LRHS_train_gt, LRHS_test_patch, LRHS_test_gt, PAN_test_patch = load_data('Pavia')
dim_input, H, W = data_orig_norm.shape[2], PAN_train_patch.shape[1], PAN_train_patch.shape[1]
print('train_sample:', LRHS_train_gt.shape[0])
####################################获得全图的输入集
LRHS_test_all, LRHS_loc = cutimg(LRHS,int(H/4),0, flag=0)
PAN_test_all, _ = cutimg(PAN,H,0, flag=1)
LRHS_test_all = np.array(LRHS_test_all)
PAN_test_all = np.expand_dims(np.array(PAN_test_all),3)
#####################################################
LRHS_test_all = LRHS_test_all.transpose([0, 3, 1, 2])
PAN_test_all = PAN_test_all.transpose([0, 3, 1, 2])
#####################################################
##################################################################################划分、打乱数据集
ratio = int(LRHS_train_gt.shape[1]/LRHS_train_patch.shape[1])
X_train_LRHS = LRHS_train_patch.transpose([0, 3, 1, 2])
X_train_PAN = PAN_train_patch
Y_train = LRHS_train_gt.transpose([0, 3, 1, 2])

X_test_LRHS = LRHS_test_patch.transpose([0, 3, 1, 2])
X_test_PAN = PAN_test_patch
Y_test = LRHS_test_gt.transpose([0, 3, 1, 2])
dim_LRHS = X_train_LRHS.shape[1]
dim_PAN = 1

X_train_PAN = np.expand_dims(X_train_PAN,axis= 1)
X_test_PAN = np.expand_dims(X_test_PAN, axis= 1)

##############################################################################
class Hyper_PNN(nn.Module):
    def __init__(self):
        super(Hyper_PNN, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(size=[H,W])
        self.conv1 = convolution(3, dim_input, 64)
        self.conv2 = convolution(3, 64, 64)

        self.spa_1 = convolution(3, 64+1, 64)
        self.spa_2 = convolution(3, 64, 64)
        self.spa_3 = convolution(3, 64, 64)
        self.spa_4 = convolution(3, 64, 64)
        self.spa_5 = convolution(3, 64, 64)

        self.fine_tune = nn.Sequential(
            convolution(3, 64, dim_input),
            convolution(3, dim_input, dim_input),
        )
        self.loss =  torch.nn.MSELoss()

    def train(self, x_LRHS,x_PAN):
        x = self.upsample(x_LRHS)
        conv = self.conv1(x)
        conv1 = self.conv2(conv)

        image_concat = torch.cat([conv1,x_PAN],dim=1)
        spa_1 = self.spa_1(image_concat)
        spa_2 = self.spa_2(spa_1)
        spa_3 = self.spa_3(spa_2)
        spa_4 = self.spa_4(spa_3)
        spa_5 = self.spa_5(spa_4)

        spe_4 = self.fine_tune(spa_5)

        return spe_4

    def forward(self, x_LRHS,x_PAN, gt=None):
        output = self.train(x_LRHS,x_PAN)
        if gt is None:
            return output
        else:
            pre_loss = torch.sqrt(self.loss(output, gt))
        return pre_loss
from torch.autograd import Variable
from torchvision import models

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def get_recon_patch(X_test_LRHS, X_test_PAN):
    num = np.shape(X_test_LRHS)[0]
    index_all = 0
    step_ = 2
    y_pred = []
    flag = 0
    while index_all<num:
        if index_all + step_ > num:
            x1, x2 = X_test_LRHS[index_all:, :, :, :], X_test_PAN[index_all:, :, :]
        else:
            x1, x2 = X_test_LRHS[index_all:(index_all+step_), :, :, :], X_test_PAN[index_all:(index_all+step_), :, :]
        index_all += step_
        x_1 = torch.tensor(x1, dtype=torch.float32).to(device)
        x_2 = torch.tensor(x2, dtype=torch.float32).to(device)
        temp1 = Hyper_PNN(x_1,x_2).cpu().detach().numpy()
        if flag == 0:
            y_pred = temp1
            flag = flag + 1
        else:
            y_pred = np.vstack((y_pred,temp1))

    return y_pred
###############################################################################
def get_recon_imag():
    y_pr_all = get_recon_patch(LRHS_test_all, PAN_test_all)
    recon_image = np.zeros((dim_input, data_orig_norm.shape[0], data_orig_norm.shape[1]))
    m = 0
    for i in range(0, recon_image.shape[1], H):
        for j in range(0, recon_image.shape[2], H):
            recon_image[:, i:(i + H), j:(j + H)] = y_pr_all[m]
            m = m + 1
    recon_image = recon_image.transpose([1, 2, 0])
    pca = PCA(n_components=dim_input_class)
    recon_image = np.float32(recon_image)
    recon_image = pca.fit_transform(recon_image.reshape(recon_image.shape[0] * recon_image.shape[1], -1))
    recon_image = recon_image.reshape(data_norm.shape[0], data_norm.shape[1], -1)
    recon_image = windowFeature(recon_image, train_loc_class, w)
    recon_image = recon_image.transpose([0, 3, 1, 2])
    return  recon_image

def get_recon_imag4evaluate():
    y_pr_all = get_recon_patch(LRHS_test_all, PAN_test_all)
    recon_image = np.zeros((data_orig_norm.shape[2], data_orig_norm.shape[0], data_orig_norm.shape[1]))
    m = 0
    for i in range(0, recon_image.shape[1], H):
        for j in range(0, recon_image.shape[2], H):
            recon_image[:, i:(i + H), j:(j + H)] = y_pr_all[m]
            m = m + 1
    recon_image_trans = recon_image.transpose([1, 2, 0])
    pca = PCA(n_components=dim_input_class)
    recon_img_PCA = pca.fit_transform(recon_image_trans.reshape(recon_image_trans.shape[0] * recon_image_trans.shape[1], -1))
    recon_img_PCA = recon_img_PCA.reshape(recon_image_trans.shape[0], recon_image_trans.shape[1], -1)
    X_train_pansharp = windowFeature(recon_img_PCA, train_loc_class, w)
    X_test_pansharp = windowFeature(recon_img_PCA, test_loc_class, w)
    X_train_pansharp = X_train_pansharp.transpose([0, 3, 1, 2])
    X_test_pansharp = X_test_pansharp.transpose([0, 3, 1, 2])
    return recon_image.transpose([1, 2, 0]), recon_img_PCA, X_train_pansharp,X_test_pansharp

def get_oa(data,X_valid_loc,Y_valid):
    size = np.shape(X_valid_loc)
    num = size[0]
    index_all = 0
    step_ = 5000
    y_pred = []
    while index_all<num:
        if index_all + step_ > num:
            input_loc = X_valid_loc[index_all:,:]
        else:
            input_loc = X_valid_loc[index_all:(index_all+step_), :]
        input = windowFeature(data,input_loc,w)
        input = torch.tensor(input.transpose([0, 3, 1, 2]), dtype=torch.float32).to(device)
        temp1,_,_ = evaluate_net(input)
        temp1 = temp1.cpu().numpy()
        y_pred1=contrary_one_hot(temp1).astype('int32')
        y_pred.extend(y_pred1)
        index_all += step_
    return y_pred

def get_index(y_pr, Y_test):
    c_sam = 0
    c_cc = 0
    c_psnr = 0
    c_ergas = 0
    c_mse = 0
    for i in range(y_pr.shape[0]):
        c_sam = c_sam + sam(y_pr[i], Y_test[i])
        c_cc = c_cc + scc(y_pr[i], Y_test[i])
        c_psnr = c_psnr + psnr(y_pr[i], Y_test[i])
        c_ergas = c_ergas + ergas(y_pr[i], Y_test[i])
        c_mse = c_mse + mse(y_pr[i] , Y_test[i])
    print('CC %f' % (c_cc/y_pr.shape[0]))
    print('SAM %f' % (c_sam/y_pr.shape[0]))
    print('MSE %f' % ( c_mse/y_pr.shape[0]))
    print('ERGAS %f' % (c_ergas/y_pr.shape[0]))
    print('PSNR %f' % (c_psnr/y_pr.shape[0]))
    print('===============================================')
    return c_cc/X_test_LRHS.shape[0], c_sam/X_test_LRHS.shape[0], c_ergas/X_test_LRHS.shape[0], c_psnr/X_test_LRHS.shape[0], c_mse/X_test_LRHS.shape[0]

def retrieve_grad(temp_optimizer):
    grad, shape, has_grad = [], [], []
    for group in temp_optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                shape.append(p.shape)
                grad.append(torch.zeros_like(p).to(p.device))
                has_grad.append(torch.zeros_like(p).to(p.device))
                continue
            shape.append(p.grad.shape)
            grad.append(p.grad.clone())
            has_grad.append(torch.ones_like(p).to(p.device))
    return grad, shape, has_grad

def unflatten_grad(grads, shapes):
    unflatten_grad, idx = [], 0
    for shape in shapes:
        length = np.prod(shape)
        unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
        idx += length
    return unflatten_grad

def flatten_grad(grads, shapes):
    flatten_grad = torch.cat([g.flatten() for g in grads])
    return flatten_grad

def project_conflicting(grads, has_grads, shapes=None):
    shared = torch.stack(has_grads).prod(0).bool()
    pc_grad, num_task = copy.deepcopy(grads), len(grads)
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
            else:
                g_j = g_j
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)

    # merged_grad[shared] = torch.stack([g[shared]
    #                                        for g in pc_grad])
    # weight_merged_grad = np.zeros((merged_grad.shape[1]))
    # for i in range(merged_grad.shape[0]):
    #     weight_merged_grad[i] = merged_grad[0,i] + merged_grad[1,i]

    merged_grad[~shared] = torch.stack([g[~shared]
                                        for g in pc_grad]).sum(dim=0)
    return merged_grad

def project_conflicting_my(grads, has_grads, shapes=None):
    shared = torch.stack(has_grads).prod(0).bool()
    pc_grad, num_task = copy.deepcopy(grads), len(grads)
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i = g_i
            else:
                g_i += (g_i_g_j) * g_j / (g_j.norm() ** 2)
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)

    merged_grad[~shared] = torch.stack([g[~shared]
                                        for g in pc_grad]).sum(dim=0)
    return merged_grad

def set_grad(grads):

    '''
    set the modified gradients to the network
    '''

    idx = 0
    for group in optimizer_1.param_groups:
        for p in group['params']:
            # if p.grad is None: continue
            p.grad = grads[idx]
            idx += 1
    return

def use_fuse_gradient_to_update(loss_Pan, loss_com):
    ###获得梯度
    grads, shapes, has_grads = [], [], []
    optimizer_1.zero_grad(set_to_none=True)
    optimizer_2.zero_grad(set_to_none=True)
    loss_Pan.backward(retain_graph=True)
    loss_com.backward(retain_graph=True)
    grads_1, shapes_1, has_grads_1 = retrieve_grad(optimizer_1)
    grads_2, shapes_2, has_grads_2 = retrieve_grad(optimizer_2)
    grads_1 = flatten_grad(grads_1, shapes_1)
    grads_2 = flatten_grad(grads_2, shapes_2)
    has_grads_1 = flatten_grad(has_grads_1, shapes_1)
    has_grads_2 = flatten_grad(has_grads_2, shapes_2)

    grads.append(grads_1)
    grads.append(grads_2)
    has_grads.append(has_grads_1)
    has_grads.append(has_grads_2)
    shapes.append(shapes_1)
    shapes.append(shapes_2)

    ###融合梯度
    pc_grad = project_conflicting_my(grads, has_grads)
    pc_grad = unflatten_grad(pc_grad, shapes[0])
    set_grad(pc_grad)
    optimizer_3.step()

    return

Hyper_PNN = Hyper_PNN()
optimizer = torch.optim.Adam(Hyper_PNN.parameters(), lr=learning_rate_PNN)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Hyper_PNN.to(device)

#####################################################################################################
from cnn_torch_grad import *
evaluate_net = torch.load('F:\wxd_第三个工作\HyperPNN_chikusei\evaluate_net.pkl')
#####################################################################################################
data_name = 'Chi'
pre = sio.loadmat('F:\wxd_第三个工作\HyperPNN_chikusei\LR_HSI_Class\Chikusei_class.mat')
data_norm = pre['data_norm']
data_norm = normalizeData(data_norm)
labels_ori = pre['labels_ori']
y_train = pre['train_y'][0]
train_loc_class = np.transpose(pre['train_loc'], [1, 0])
y_test_class = pre['test_y'][0]
test_loc_class = np.transpose(pre['test_loc'], [1, 0])
dim_out_class = int(np.max(labels_ori))  # 类别数
Y_test_class = y_test_class
from sklearn.decomposition import PCA
pca = PCA(n_components=dim_input_class)
orig_img_PCA = pca.fit_transform(data_norm.reshape(data_norm.shape[0] * data_norm.shape[1], -1))
orig_img_PCA = orig_img_PCA.reshape(data_norm.shape[0], data_norm.shape[1], -1)


# 转换数据维度
X_train_orig = windowFeature(orig_img_PCA, train_loc_class, w)
X_train_orig= X_train_orig.transpose([0, 3, 1, 2])
Y_train_orig = y_train
#####################################################################################################

optimizer_1 = torch.optim.Adam(Hyper_PNN.fine_tune.parameters(), lr= 0.0001)
optimizer_3 = torch.optim.Adam(Hyper_PNN.fine_tune.parameters(), lr= 0.000001)
optimizer_2 = torch.optim.Adam(evaluate_net.grad_network.parameters(), lr=0.0001)

num_epoch_in_fine_tune = 200
batch_size_fine_tune = 128
epoch = 0
print('===================phase==============================')
while epoch < num_epoch_PNN:
    batch_LRHS, batch_PAN, batch_y = next_batch_pansharp(X_train_LRHS, X_train_PAN, Y_train, index, batch_size)
    batch_LRHS = torch.tensor(batch_LRHS, dtype=torch.float32).to(device)
    batch_PAN = torch.tensor(batch_PAN, dtype=torch.float32).to(device)
    batch_y = torch.from_numpy(batch_y.astype(np.float32)).to(device)

    optimizer.zero_grad()
    loss = Hyper_PNN(batch_LRHS, batch_PAN, batch_y)
    loss.backward()
    optimizer.step()  # 更新分类层权值
    # print(loss)
    index = index + batch_size
    step += 1

    if index > (X_train_LRHS.shape[0]):
        index = batch_size
        index_train = np.arange(X_train_LRHS.shape[0])
        np.random.shuffle(index_train)
        X_train_LRHS, X_train_PAN, Y_train = X_train_LRHS[index_train, :, :, :], X_train_PAN[index_train, :,
                                                                                 :], Y_train[index_train, :, :, :]
        epoch = epoch + 1

print('======================= phase 1 finished ========================')


y_pr = get_recon_patch(X_test_LRHS, X_test_PAN)  ##获得每一个图像块
c_cc, c_sam, c_ergas, c_psnr, epsilon = get_index(y_pr, Y_test)  # 评估锐化后的每个块

flag_count = 0
flag = 500
num_epoch_in_fine_tune_count = 0
if (flag_count < flag):
    while (num_epoch_in_fine_tune_count < num_epoch_in_fine_tune):
        X_train_pansharp = get_recon_imag()
        index_fine_tune = batch_size_fine_tune
        while num_epoch_in_fine_tune_count < num_epoch_in_fine_tune:
            batch_x_pansharp = torch.tensor(next_batchx(X_train_pansharp, index_fine_tune), dtype=torch.float32).to(device)
            batch_x_orig = torch.tensor(next_batchx(X_train_orig, index_fine_tune), dtype=torch.float32).to(device)
            batch_y_orig = torch.tensor(next_batchy(Y_train_orig, index_fine_tune), dtype=torch.float32).to(device)
            fea_pansharp, _, _ = evaluate_net(batch_x_pansharp)
            fea_orig, _, _ = evaluate_net(batch_x_orig)
            loss_class = nn.CrossEntropyLoss()
            # loss_compare = torch.nn.L1Loss()

            #######loss形式
            # optimizer_1.zero_grad()
            # loss_Pan = Hyper_PNN(batch_LRHS, batch_PAN, batch_y)
            # # loss_all = loss_Pan+ loss_class(fea_orig, batch_y_orig.long()-1)
            # loss_all.backward()
            # optimizer_1.step()  # 更新分类层权值
            ######梯度形式
            loss_Pan = Hyper_PNN(batch_LRHS, batch_PAN, batch_y)
            loss_cla = loss_class(fea_orig, batch_y_orig.long()-1)
            use_fuse_gradient_to_update(loss_Pan,loss_cla)


            index_fine_tune = index_fine_tune + batch_size_fine_tune
            if index_fine_tune > train_loc_class.shape[0]:
                index_fine_tune = batch_size_fine_tune
                index_train = np.arange(X_train_pansharp.shape[0])
                np.random.shuffle(index_train)
                X_train_pansharp = X_train_pansharp[index_train, :, :, :]
                Y_train_orig = Y_train_orig[index_train]
                num_epoch_in_fine_tune_count = num_epoch_in_fine_tune_count + 1
                flag_count = flag_count + 1
print('======================= epoch=1500 ========================')
y_pr = get_recon_patch(X_test_LRHS, X_test_PAN)
c_cc, c_sam, c_ergas, c_psnr, epsilon = get_index(y_pr, Y_test)

print("Optimization Finished!")

"""eval"""
print('evaling...')

data_flag = 1
fuse = get_recon_patch(X_test_LRHS,X_test_PAN)
for i in range(X_test_LRHS.shape[0]):
    fake = np.squeeze(fuse[i])
    fake = fake.transpose([1, 2,0])
    cv2.imwrite(str(i) + '.png', show_hsi(fake, data_flag))
sio.savemat('HR_HSI.mat', {'fuse': fake})
################################################################################################################################
'evaluate the classification performance of pansharpening image'
import scipy.io as sio
import os
from pre2 import normalizeData

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        # conv   = self.bn(conv)
        relu = self.relu(conv)
        return relu

class CNN(nn.Module):
    def __init__(self, dim_input, dim_out_class):
        super(CNN, self).__init__()

        self.conv1 = convolution(3, 27, 32)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2, padding=1)
        self.conv2 = convolution(3, 32, 64)
        self.pool2 = nn.MaxPool2d((2, 2), stride=2, padding=1)
        self.grad_network = nn.Sequential(
            convolution(3, 64, 64),
            convolution(3, 64, 64),
            convolution(3, 64, dim_input),
            convolution(3, dim_input, dim_input)
        )
        self.fc = nn.Linear(8192, dim_out_class)
        self.cross_loss = nn.CrossEntropyLoss()
        self.contrast = nn.Linear(8192, 1024)
        self.con_fea = nn.Linear(1024, 128)

        self.Contrast_loss = Contrast_loss()

    def train(self, x):

        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv_vec = self.grad_network(pool2)
        conv5_vec = conv_vec.reshape(-1, conv_vec.shape[1] * conv_vec.shape[2] * conv_vec.shape[3])
        fc1 = self.fc(conv5_vec)
        contrast = self.contrast(conv5_vec)
        con_fea = self.con_fea(contrast)

        return fc1,con_fea,conv5_vec

    def forward(self, x, gt=None):
        output = self.train(x)
        if gt is None:
            return output
        else:
            pre_loss = self.cross_loss(output[0], gt.long()-1)
        return pre_loss

np.random.seed(1126)
dim_input = 27
step = 1
batch_size = 64
w = 27
num_epoch = 500


data_name = 'Pavia'  # 'Indian_pines'#'PaviaU'#'Washington'#'Salinas'
_, recon_img_PCA, X_train_pansharp,X_test_pansharp = get_recon_imag4evaluate()
cv2.imwrite('recon_img.png', show_hsi(_, data_flag))
# cv2.imwrite('recon_img_pca.png', show_hsi(recon_img_PCA, data_flag))

X_train = X_train_pansharp
X_test = X_test_pansharp
Y_train = y_train
Y_test = Y_test_class
# 转换数据维度

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

# 打乱顺序
index_train = np.arange(X_train.shape[0])
np.random.shuffle(index_train)
X_train = X_train[index_train, :, :, :]
Y_train = Y_train[index_train]

net = CNN(dim_input,dim_out_class)
optimizer_after_class = torch.optim.Adam(net.parameters(), lr= 0.0001)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
net.to(device)

#############training###################################################

epoch = 0
index = batch_size
step = 1

while epoch < num_epoch:
    batch_x = torch.tensor(next_batchx(X_train, index), dtype=torch.float32).to(device)
    batch_y = torch.from_numpy(next_batchy(Y_train, index).astype(np.float32)).to(device)

    optimizer_after_class.zero_grad()
    loss = net(batch_x, batch_y)
    loss.backward()
    optimizer_after_class.step()  # 更新分类层权值
    index = index + batch_size
    # print('loss_all:{:.3f}'.format(loss.data))

    if index > train_loc_class.shape[0]:
        index = batch_size
        index_train = np.arange(X_train.shape[0])
        np.random.shuffle(index_train)
        X_train = X_train[index_train,:,:,:]
        Y_train = Y_train[index_train]
        epoch = epoch + 1


print('Finished Training')

def get_oa(data,X_valid_loc,Y_valid):
    size = np.shape(X_valid_loc)
    num = size[0]
    index_all = 0
    step_ = 5000
    y_pred = []
    while index_all<num:
        if index_all + step_ > num:
            input_loc = X_valid_loc[index_all:,:]
        else:
            input_loc = X_valid_loc[index_all:(index_all+step_), :]
        input = windowFeature(data,input_loc,w)
        input = torch.tensor(input.transpose([0, 3, 1, 2]), dtype=torch.float32).to(device)
        temp1,_,_ = net(input)
        temp1 = temp1.cpu().numpy()
        y_pred1=contrary_one_hot(temp1).astype('int32')
        y_pred.extend(y_pred1)
        index_all += step_
    return y_pred

with torch.no_grad():
    y_pr = get_oa(recon_img_PCA,test_loc_class,Y_test_class)
    y_real = Y_test_class
    oa = accuracy_score(y_real, y_pr)
    per_class_acc = recall_score(y_real, y_pr, average=None)
    aa = np.mean(per_class_acc)
    kappa = cohen_kappa_score(y_real, y_pr)
    print(per_class_acc)
    print(oa, aa, kappa)





