import torch
import torch.nn as nn
import torch.nn.functional
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math

# this configures the device to use cuda if it is available , else use the cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# basic parameters for solving the problem
epochs = 10
batch_size = 7
rate = 3e-4

#here we are normalising the data to maximise accuracy, why3*3 tuples? because there are 3 dimensions RGB
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)
                                    


class resblock(nn.Module):
	def __init__(self,in_channels,out_channels,identity_downsample=None,stride=1):
		super(resblock,self).__init__()
		self.expansion=4
		#size increasses by 4
		self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
		self.n1=nn.BatchNorm2d(out_channels)
		self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
		self.bn2=nn.BatchNorm2d(out_channels)
		self.conv3=nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,stride=1,padding=0)
		self.bn3=nn.BatchNorm2d(out_channels*self.expansion)
		self.relu=nn.ReLU()
		self.identity_downsample=identity_downsample
	def forward(self,x):
		identity=x 
		x = self.conv1(x)
		x=self.bn1(x)
		x=self.relu(x)
		x=self.conv2(x)
		x=self.bn2(x)
		x=self.relu(x)
		x=self.conv3(x)
		x=self.bn3(x)
		if self.identity_downsample is not None:
			identity=self.identity_downsample(identity)
			
		x+=identity
		x=self.relu(x)
		return x


class ResNet(nn.Module):#[3,4,6,3]
	def __init__(self,block,layers,image_channels,num_classes):
		super(ResNet,self).__init__()
		#image_channels=3,RGB,numclasses=10 in CIFAR10
		self.in_channels=64
		self.conv1=nn.Conv2d(image_channels,64,kernel_size=7,stride=2,padding=3)
		self.bn1=nn.BatchNorm2d(64)
		self.relu=nn.ReLU()
		self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		#Resnet layers nowm this is the basic layer
		self.layer1=self.make_layer(resblock,layers[0],out_channels=64,stride=1)
		self.layer2=self.make_layer(resblock,layers[1],out_channels=128,stride=2)
		self.layer3=self.make_layer(resblock,layers[2],out_channels=256,stride=2)
		self.layer4=self.make_layer(resblock,layers[3],out_channels=512,stride=2)
		self.avgpool=nn.AvgPool2d((1,1))
		self.fc=nn.Linear(512*4,num_classes)

	def make_layer(self,block,num_residual_block,out_channels,stride):
		identity_downsample=None 
		residual_layer=[]
		if(stride!=1 or self.in_channels!=out_channels*4):
			identity_downsample=nn.Sequential(nn.Conv2d(self.in_channels,out_channels*4,kernel_size=1,stride=stride),nn.BatchNorm2d(out_channels*4))
		residual_layer.append(block(self.in_channels,out_channels,identity_downsample,stride))	
		self.in_channels=out_channels*4
		for i in range(num_residual_block-1):
			#we computed one already
			residual_layer.append(block(self.in_channels,out_channels))  
			#256 is the in channels, now we need to converrt 256 to 64 then back, 64*4, input 256, output 256 
			#
		nn.Sequential(*residual_layer)
	def  forward(self,x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
  
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = x.reshape(x.shape[0],-1)
		x = self.fc(x)
		return x 

def ResNet50(image_channels=3,num_classes=10):
	return ResNet(resblock,[3,4,6,3],image_channels,num_classes)
def ResNet101(image_channels=3,num_classes=10):
	return ResNet(resblock,[3,4,23,3],image_channels,num_classes)


model = ResNet50().to(device)
cross_entropy = nn.CrossEntropyLoss()
#cross entropy is a type of non-linear(logarithmic to be precise) log function.
optimizer = torch.optim.SGD(model.parameters(),lr=rate)



n_total_steps = len(train_loader)
for epoch in range(epochs):
    for images, labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs = model(images)
        loss = cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


print('Finished Training')

with torch.no_grad():
    n_correct = 0
    n_wrong = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        temp, predicted = torch.max(outputs, 1)
        n_wrong+=(predicted!=labels).sum().item()
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / (n_wrong+n_correct)
    print(f'Accuracy of the network: {acc} %')