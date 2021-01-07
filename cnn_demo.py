import torchvision
import torch
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
data_train = datasets.MNIST(
    root = './',
    transform = transform, #可以在数据集较少是做诸如Data Argumentation操作等
    train=True,
    download=True,
)
data_test = datasets.MNIST(
    root='./',
    transform=transform,
    train=False,
)
data_loader_train = torch.utils.data.DataLoader(
    dataset = data_train,
    batch_size = 64,  #设置每批装载的数据图片为64个
    shuffle = True,  #装载过程中随机乱序
    num_workers=0,
)
data_loader_test = torch.utils.data.DataLoader(
    dataset = data_test,
    batch_size = 64,
    shuffle = True,
    num_workers=0,
)

class Cnn(torch.nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = torch.nn.Sequential(
            #64x1x28x28
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), #input image channel 为1，输出channel为64，步长为1，填白为1
            #64x64x28x28 28=28+2-1
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            #64x128x28x28 28=28+2-1
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
            #64x128x14x14 14=(28-2)/2+1
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5), #防止模型过拟合
            torch.nn.Linear(1024, 10),
        )

    def forward(self,x):
        #print(x.size())
        x = self.conv1(x)
        #print(x.size())
        x = x.view(-1, 14*14*128) #压缩扁平化处理，直接压缩？
        #print(x.size())
        x = self.dense(x)
        return x

def test():
    data_loader_test = torch.utils.data.DataLoader(
        dataset=data_test,
        batch_size=16,
        shuffle=True,
    )
    X_test, y_test = next(iter(data_loader_test))
    inputs = Variable(X_test)
    cnn = Cnn()
    cnn.load_state_dict(torch.load('model_parameter.pkl'))
    params=cnn.state_dict() 
    for k,v in params.items():
        print(k, v)    #打印网络中的变量名

    pred = cnn(inputs)
    #print(pred)lin
    _, pred = torch.max(pred, 1) #pylint: disable=no-member
    print(pred)
    print(y_test)
    img = torchvision.utils.make_grid(X_test)
    img = img.numpy().transpose(1,2,0)
    std = [0.5]
    mean = [0.5]
    img = img*std+mean
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    cnn = Cnn()
    # print(cnn)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters())
    n_epochs = 5
    #cnn.load_state_dict(torch.load('model_parameter.pkl'))
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(epoch, n_epochs))
        for data in data_loader_train:
            X_train, y_train = data
            X_train, y_train = Variable(X_train), Variable(y_train)
            outputs = cnn(X_train)
            _,pred = torch.max(outputs.data, 1) #pylint: disable=no-member
            optimizer.zero_grad()
            loss = cost(outputs, y_train)

            loss.backward()
            optimizer.step()
            #print(loss.data)
            running_loss += loss.data
            running_correct += torch.sum(pred==y_train.data) #pylint: disable=no-member
        testing_correct = 0
        for data in data_loader_test:
            X_test, y_test = data
            X_test, y_test = Variable(X_test), Variable(y_test)
            outputs = cnn(X_test)
            _, pred = torch.max(outputs.data, 1) #pylint: disable=no-member
            testing_correct += torch.sum(pred == y_test.data) #pylint: disable=no-member
        print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss/len(data_train),
                                                                                      100*running_correct/len(data_train),
                                                                                      100*testing_correct/len(data_test)))    
    torch.save(cnn.state_dict(), 'model_parameter.pkl')
    test()