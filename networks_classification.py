import torch.nn as nn
from torchvision import models

class FeatureNet(nn.Module):

	def __init__(self):
		
		super(FeatureNet, self).__init__()
		resnet50 = models.resnet50(pretrained = True, progress = True)
		resnet50 = list(resnet50.children())[:-1]
		self.conv_layers = nn.Sequential(*resnet50)
		self.fc_layer1 = nn.Sequential(
			nn.Linear(2048, 128),
			nn.BatchNorm1d(128),
			nn.ReLU()
			)
		self.fc_layer2 = nn.Linear(128, 751)
		# self.fc_layer2 = nn.Linear(256, 1376)

	def forward(self, x):
		
		x = self.conv_layers(x)
		x = x.view(x.shape[0], -1)
		x = self.fc_layer1(x)
		x = self.fc_layer2(x)
		
		return x

	def get_features(self, x):
		
		return self.forward(x)

	def get_embeddings(self, x):

		x = self.conv_layers(x)
		x = x.view(x.shape[0], -1)
		x = self.fc_layer1(x)

		return x
