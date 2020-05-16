#-*- coding: utf-8 -*
'''
配置说明：
1. 原位深遥感图像，通过逐像素点除以2047（即2^11)的方式归一化，制作成网络输入，将网络输出保存成图像前，再乘以2047，
复原到原位深图像
2. GF：trainset: 19976 ×（128×128），testset: 90 ×（512×512）
3. QB：trainset: 18123 ×（128×128），testset: 24 ×（512×512）

5. SGD，momentum=0.9， learning_rate 前2层：0.0001，第3层：0.00001
6. MSE loss
7. total epochs： 600， batchsize：128， total_iterations: 1125000
'''
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
from torch.nn import init
import time
import scipy.io as sio
import gdal, ogr, os, osr
from os.path import join
from visdom import Visdom

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
## 超参数设置
version = 1   # 版本号
mav_value = 1023  #  GF:1023  QB:2047
satellite = 'gf'  #  gf，qb
method = 'pnn'
train_batch_size = 128
test_batch_size = 1
total_epochs = 600
lr = 0.0001
test_freq = 20
model_backup_freq = 20
num_workers = 1

## 文件夹设置
traindata_dir = '../TIF/train/'
testdata_dir = '../TIF/test/'
testsample_dir = '../pnn-results/test-samples-v{}/'.format(version)  # 保存测试阶段G生成的图片
evalsample_dir = '../pnn-results/eval-samples-v{}/'.format(version)
record_dir = '../pnn-results/record-v{}/'.format(version)  # 保存训练阶段的损失值
model_dir =  '../pnn-results/models-v{}/'.format(version)
backup_model_dir = join(model_dir, 'backup_model/')
checkpoint_model = join(model_dir, '{}-{}-model.pth'.format(satellite, method))

## 创建文件夹
if not os.path.exists(evalsample_dir):
    os.makedirs(evalsample_dir)
if not os.path.exists(testsample_dir):
    os.makedirs(testsample_dir)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(backup_model_dir):
    os.makedirs(backup_model_dir)

## Device configuration
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('==> gpu or cpu:', device, ', how many gpus available:', torch.cuda.device_count())

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ["mul.tif"])

def load_image(filepath):
    img = gdal.Open(filepath)  # 原始数据
    img = img.ReadAsArray()  # [C,W,H]
    if filepath.split('_')[1] != 'pan.tif':
        img = img.transpose(1, 2, 0)  # [W,H,C]
    img = img.astype(np.float32) / mav_value    # 归一化处理
    return img

class DatasetFromFolder(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_filenames = [join(img_dir, x.split('_')[0]) for x in os.listdir(img_dir) if is_image_file(x)]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):   # idx的范围是从0到len（self）
        input_pan = load_image('%s_pan.tif'%self.image_filenames[index])
        input_lr_u = load_image('%s_lr_u.tif'%self.image_filenames[index])
        target = load_image('%s_mul.tif'%self.image_filenames[index])

        if self.transform:
            input_pan = self.transform(input_pan)
            input_lr_u = self.transform(input_lr_u)
            target = self.transform(target)
        return input_pan, input_lr_u, target

class ToTensor(object):
    def __call__(self, input):
        if input.ndim == 3:
            input = np.transpose(input, (2, 0, 1))
            input = torch.from_numpy(input).type(torch.FloatTensor)
        else:
            input = torch.from_numpy(input).unsqueeze(0).type(torch.FloatTensor)
        return input

def get_train_set(traindata_dir):
    return DatasetFromFolder(traindata_dir,
                             transform = transforms.Compose([ToTensor()]))
def get_test_set(testdata_dir):
    return DatasetFromFolder(testdata_dir,
                             transform = transforms.Compose([ToTensor()]))

transformed_trainset = get_train_set(traindata_dir)
transformed_testset = get_test_set(testdata_dir)

## 训练集  ## 验证集  ## 测试集
trainset_dataloader = DataLoader(dataset=transformed_trainset, batch_size=train_batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory= True, drop_last=True)
testset_dataloader = DataLoader(dataset=transformed_testset, batch_size=test_batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, drop_last=True)

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.opt = opt
        self.stream = torch.cuda.Stream()
        self.preload()
    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

class PNN(nn.Module):
    def __init__(self):
        super(PNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=7, out_channels=48, kernel_size=9, stride=1, padding=4)
        self.conv_2 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        fea = self.relu(self.conv_1(x))   #  [batch_size,7,128,128]
        fea =  self.relu(self.conv_2(fea))
        out = self.conv_3(fea)
        return out

def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, bandSize):
    if (bandSize == 4):
        cols = array.shape[2]
        rows = array.shape[1]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')   # #存的数据格式

        outRaster = driver.Create(newRasterfn, cols, rows, 4, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        for i in range(1, 5):
            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(array[i - 1, :, :])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
    elif (bandSize == 1):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array)

def denorm(x):
    x = (x * mav_value).astype(np.uint16)
    return x

def eval_img_save(x,name,k):
    x = x.numpy()
    x = np.transpose(x, (0, 2, 3, 1))   # [batch_size,512,512,4]
    if name == 'real_images':
        array2raster(join(evalsample_dir, 'real_images_{}_epoch{}.tif'.format(k + 1,total_epochs)),
                     [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)
    else:
        array2raster(join(evalsample_dir, '{}_v{}_eval_fused_images_{}_epoch{}.tif'.format(method, version, k + 1,total_epochs)),
                     [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)

def test_img_save(x,name,epoch):
    x = np.transpose(x, (0, 2, 3, 1))
    x = x.numpy()  # [batch_size,512,512,4]
    if name == 'test_fused_images':
        array2raster(join(testsample_dir, 'test_fused_images_9_epoch{}.tif'.format(epoch)),
                            [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)
    elif name == 'real_images':
        array2raster(join(testsample_dir, 'real_images_9_epoch{}.tif'.format(epoch)),
                            [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)
    elif name == 'test_pan_images':
        array2raster(join(testsample_dir, 'test_pan_images_9_epoch{}.tif'.format(epoch)),
                     [0, 0], 8, 8, denorm(x[0].reshape(x.shape[1], x.shape[2])), 1)
    else:
        array2raster(join(testsample_dir, 'test_lrms_images_9_epoch{}.tif'.format(epoch)),
                            [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)

criterion = nn.MSELoss().to(device)
model = PNN()
model.to(device)

# 前2层lr，第3层lr * 0.1
conv_3_params = list(map(id, model.conv_3.parameters()))
base_params = filter(lambda p: id(p) not in conv_3_params,
                     model.parameters())
optimizer = torch.optim.SGD([{'params': base_params},
                    {'params': model.conv_3.parameters(), 'lr': lr * 0.1}],lr=lr, momentum=0.9)

if (torch.cuda.device_count() > 1):
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# 模型训练
def train(model, trainset_dataloader, start_epoch):
    print('===>Begin Training!')
    model.train()
    steps_per_epoch = len(trainset_dataloader)
    total_iterations = total_epochs * steps_per_epoch
    print('total_iterations:{}'.format(total_iterations))

    train_loss_record = open('%s/train_loss_record.txt' % record_dir, "w")
    epoch_time_record = open('%s/epoch_time_record.txt' % record_dir, "w")
    time_sum = 0

    viz = Visdom()
    viz.line(np.array([0.]), np.array([0.]), win='pnn_train_loss', opts=dict(title='pnn_train loss'))

    for epoch in range(start_epoch + 1, total_epochs + 1):
        start = time.time()  # 记录每轮训练的开始时刻
        prefetcher = DataPrefetcher(trainset_dataloader)
        data = prefetcher.next()
        i = 0
        while data is not None:
            i += 1
            if i >= iters_per_epoch:
                break
            img_pan, img_lr_u, target = data[0].to(device), data[1].to(device), data[2].to(device)  # cuda tensor [batchsize,C,W,H]
            NDWI = ((img_lr_u[:, 1, :, :] - img_lr_u[:, 3, :, :]) / (img_lr_u[:, 1, :, :] + img_lr_u[:, 3, :, :])).unsqueeze(1)
            NDVI = ((img_lr_u[:, 3, :, :] - img_lr_u[:, 2, :, :]) / (img_lr_u[:, 3, :, :] + img_lr_u[:, 2, :, :])).unsqueeze(1)
            input_joint = torch.cat([img_lr_u, img_pan, NDWI, NDVI], dim=1)  # [batch_size,5+2,128,128]

            train_fused_images = model(input_joint)  # 网络输出
            train_loss = criterion(train_fused_images, target)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        data = prefetcher.next()

        print('=> {}-{}-Epoch[{}/{}]: train_loss: {:.15f}'.format(satellite, method, epoch, total_epochs, train_loss.item()))
        train_loss_record.write("Epoch[{}/{}]: train_loss: {:.15f}\n".format(epoch, total_epochs, train_loss.item()))

        viz.line(np.array([train_loss.item()]), np.array([epoch]), win='pnn_train_loss', update='append')

        # Save the model checkpoints and backup
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, checkpoint_model)

        # backup a model every epoch
        if epoch % model_backup_freq == 0:
            torch.save(model.state_dict(), join(backup_model_dir, '{}-{}-model-epochs{}.pth'.format(satellite, method, epoch)))

        if epoch % test_freq == 0:
            checkpoint = torch.load(checkpoint_model)
            model.load_state_dict(checkpoint['model'])
            print('==>Testing the model after training {} epochs'.format(epoch))
            test(model,testset_dataloader, epoch)

        # 输出每轮训练花费时间
        time_epoch = (time.time() - start)
        time_sum += time_epoch
        print('==>No:{} epoch training costs {:.4f}min'.format(epoch, time_epoch / 60))
        epoch_time_record.write(
            "No:{} epoch training costs {:.4f}min\n".format(epoch, time_epoch / 60))

def test(model, testset_dataloader,epoch):
    avg_test_loss = 0
    model.eval()
    test_loss_record = open('%s/test_loss_record.txt' % record_dir, "a")
    with torch.no_grad():
        for k, data in enumerate(testset_dataloader):
            img_pan, img_lr_u, target = data[0].to(device), data[1].to(device), data[2].to(device)  # 此时数据类型是cuda tensor [batchsize,C,W,H]
            NDWI = ((img_lr_u[:, 1, :, :] - img_lr_u[:, 3, :, :]) / (img_lr_u[:, 1, :, :] + img_lr_u[:, 3, :, :])).unsqueeze(1)
            NDVI = ((img_lr_u[:, 3, :, :] - img_lr_u[:, 2, :, :]) / (img_lr_u[:, 3, :, :] + img_lr_u[:, 2, :, :])).unsqueeze(1)
            input_joint = torch.cat([img_lr_u, img_pan, NDWI, NDVI], dim=1)  # [batch_size,5+2,128,128]
            test_fused_images = model(input_joint)  # # cuda tensor [bs,1,W,H]
            # 损失函数
            test_loss = criterion(test_fused_images,target)
            avg_test_loss += test_loss.item()

            # 保存融合图像
            if k == 8:
                print('==>Save the test_fused_images')
                test_fused_images = test_fused_images.cpu()
                test_img_save(test_fused_images, 'test_fused_images', epoch)

                if epoch == test_freq:
                    print('==>Save the reference_images')
                    real_images, img_lr_u, img_pan = target.cpu(), img_lr_u.cpu(), img_pan.cpu()
                    test_img_save(real_images, 'real_images', epoch)
                    test_img_save(img_lr_u, 'test_lrms_images', epoch)
                    test_img_save(img_pan, 'test_pan_images', epoch)

        print("===>Epoch{} Avg.test.loss: {:.10f} ".format(epoch, avg_test_loss / len(testset_dataloader)))
        test_loss_record.write("Epoch{} Avg.test.loss: {:.10f}\n".format(epoch, avg_test_loss / len(testset_dataloader)))
        test_loss_record.close()

def eval(model, testset_dataloader):
    model.eval()
    eval_loss_record = open('%s/eval_loss_record.txt' % record_dir, "w")
    with torch.no_grad():
        for k, data in enumerate(testset_dataloader):
            img_pan, img_lr_u, target = data[0].to(device), data[1].to(device), data[2].to(device)  # 此时数据类型是cuda tensor [batchsize,C,W,H]
            NDWI = ((img_lr_u[:, 1, :, :] - img_lr_u[:, 3, :, :]) / (img_lr_u[:, 1, :, :] + img_lr_u[:, 3, :, :])).unsqueeze(1)
            NDVI = ((img_lr_u[:, 3, :, :] - img_lr_u[:, 2, :, :]) / (img_lr_u[:, 3, :, :] + img_lr_u[:, 2, :, :])).unsqueeze(1)
            input_joint = torch.cat([img_lr_u, img_pan, NDWI, NDVI], dim=1)  # [batch_size,5+2,128,128]
            eval_fused_images = model(input_joint)  # # cuda tensor [bs,1,W,H]
            # 损失函数
            eval_loss = criterion(eval_fused_images,target)
            print("===>Batch:{} Eval.loss: {:.10f} ".format(k+1, eval_loss.item()))
            eval_loss_record.write("Batch:{} Eval.loss: {:.10f}\n".format(k+1, eval_loss.item()))

            # 保存融合图像
            print('==>Save the fused_images')
            eval_fused_images, real_images = eval_fused_images.cpu(), target.cpu()
            eval_img_save(eval_fused_images, 'eval_fused_images', k)
            eval_img_save(real_images, 'real_images', k)

    eval_loss_record.close()

def main():
    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(checkpoint_model):
        print("==> loading checkpoint '{}'".format(checkpoint_model))
        checkpoint = torch.load(checkpoint_model)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('==> 加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('==> 无保存模型，将从头开始训练！')

    train(model, trainset_dataloader, start_epoch)

    eval(model, testset_dataloader)

if __name__ == '__main__':
    main()

