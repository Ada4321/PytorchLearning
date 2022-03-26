#导入相关库
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.nn import functional as f
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#定义hyperparameter（能够自行定义的参数）
batch_size = 128 #每批处理数据的数量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 50      #训练轮次

#构建transforms，处理图像
pipeline = transforms.Compose([
    transforms.ToTensor(),  #将图片转化为tensor对象
    transforms.Normalize((0.1307,),(0.3081,))  #正则化，过拟合时降低复杂度
])

#下载数据集
train_set = datasets.MNIST('data',train=True,download=True,transform=pipeline)
test_set = datasets.MNIST('data',train=False,download=False,transform=pipeline)
#加载数据集
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True) #shuffle：打乱顺序
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True)

#所有分类
classes = ('0','1','2','3','4','5','6','7','8','9')

# 构建网络模型
class HandwritingDecypher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  # 1:灰度图片的通道 10:输出通道 5:kernel
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 输入通道，输出通道
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):  # 前向传播
        input_size = x.size(0)  # input_size = batch_size(x.size = batch_size*1*28*28)
        x = self.conv1(x)
        x = f.relu(x)  # 激活函数：ReLU
        x = f.max_pool2d(x, 2, 2)  # 池化层，对图片进行压缩——图片转化为分块矩阵，提取最大的分块
        x = self.conv2(x)
        x = f.relu(x)
        # 拉平，立体->平面，送入全联接层
        x = x.view(input_size, -1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        # 计算概率
        output = f.log_softmax(x, dim=1)
        return output

#定义优化器
model = HandwritingDecypher().to(device)
optimizer = optim.Adam(model.parameters())

#初始化tensorboard
writer = SummaryWriter('./runs./HandwritingDecypher')

#网格化记录图片
dataiter = iter(train_loader)  #生成迭代器
images,labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)  #网格化

#tensorboard添加images
writer.add_image('handwriting_images',img_grid)

#启动tensorboard：在runs上层文件夹使用命令行运行  tensorboard --logdi runs

#模型可视化，tensorboard添加graphs
writer.add_graph(model,images)

#随机选取n个数据点及其标签
def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))  #返回[0,n)的随机排列
    return data[perm][:n], labels[perm][:n]

images, labels = select_n_random(train_set.data, train_set.targets)

#获取每张图片的标签
class_labels = [classes[lab] for lab in labels]

#日志嵌入
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,          #metadata:元数据，指类型标签
                    label_img=images.unsqueeze(1))  #在维度1位置插入一个size为1的维度，扩展成四个维度

#追踪模型训练过程
def images_to_probs(model, images):
   output = model(images)
   #将预测的概率转化为对应标签
   _, preds_tensor = torch.max(output, 1)
   preds = np.squeeze(preds_tensor.numpy())
   return preds, [f.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]  #返回预测结果及概率

def plot_classes_preds(model, images, label):
   preds, probs = images_to_probs(model, images)
   #绘制图像及其预测结果
   fig = plt.figure(figsize=(10, 10))
   for idx in np.arange(4):
       ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
       ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
           classes[preds[idx]],
           probs[idx] * 100.0,
           classes[label[idx]]),
                   color=("green" if preds[idx]==label[idx].item() else "red"))
   return fig

#定义训练方法
def training_model(model,device,train_loader,optimizer,epoch):
    #模型训练
    model.train()
    for batch_index,(data,label) in enumerate(train_loader):
        data,label = data.to(device),label.to(device)
        #梯度初始化为0
        optimizer.zero_grad()
        #训练后的结果
        output = model(data)
        #计算损失
        loss = f.cross_entropy(output,label)
        #找到最大概率对应下标(预测值)
        pred = output.max(1,keepdim=True)[1]  #得到每行最大值的下标索引
        #反向传播
        loss.backward()
        #参数优化
        optimizer.step()
        #写入损失
        if batch_index % 100 == 0:  #每100个打印一次
            print('Train Epoch:{}\t Loss:{:.4f}'.format(epoch,loss.item()))
            #tensorboard添加scalars
            writer.add_scalar('Loss',
                            loss / 1000,
                            ep * len(train_loader) + batch_index)
            writer.add_figure('Predictions vs. Actuals',   #添加预测vs真实值对比图
                            plot_classes_preds(model, data, label),
                            global_step=ep * len(train_loader) + batch_index)

#绘制预测准确的曲线
def add_pr_curve_tensorboard(class_index, test_probs, gt_labels, global_step=0):
    gt_labels = gt_labels == class_index
    tensorboard_probs = test_probs[:, class_index]
    writer.add_pr_curve(classes[class_index],
                        gt_labels,
                        tensorboard_probs,
                        global_step=global_step)

#定义测试方法
def test_model(model,device,test_loader):
    #模型验证
    model.eval()
    #正确率
    correct = 0.0
    #测试损失
    test_loss = 0.0
    class_probs = []
    class_preds = []
    gt_labels = []
    with torch.no_grad():  #不计算梯度，不进行反向传播
        for data,label in test_loader:
            data,label = data.to(device),label.to(device)
            images = data
            #测试结果
            output = model(data)
            class_probs_batch = [f.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)
            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)
            gt_labels.append(label)
            #计算损失
            test_loss = f.cross_entropy(output,label).item()
            #找到最大概率对应下标(预测值)
            pred = output.max(1,keepdim=True)[1]  #得到每行最大值的下标索引
            #累计正确预测数
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)
        gt_labels = torch.cat(gt_labels)
        #打印测试结果
        print('Average Loss : {:.4f}, Accuracy : {:.4f}\n'.format(test_loss,correct/len(test_loader.dataset)))
        for i in range(len(classes)):
            #逐个标签在tensorboard添加pr_curve
            add_pr_curve_tensorboard(i, test_probs, gt_labels, i)
        writer.close()

#开始训练
for ep in range(epochs):
    training_model(model,device,train_loader,optimizer,ep)
    test_model(model,device,test_loader)