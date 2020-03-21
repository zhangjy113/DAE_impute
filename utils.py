#encoding:utf-8
import torch
import os
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from math import isnan
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

class MaskLoss(nn.Module): # loss function with mask(1:base_loss, 0:no_loss)
	def __init__(self, base_loss = nn.MSELoss(reduction='none')):
		super(MaskLoss, self).__init__()
		self.base_loss = base_loss

	def forward(self, output, target, a_m):
		a = 1
		#loss = torch.sum(self.base_loss(output, target) * mask)
		loss = torch.sqrt(torch.mean(torch.pow(a_m * output - a_m * target, 2)))
		#loss = torch.sqrt(torch.mean(torch.pow(a_m * output - a_m * target, 2) * 1 + (1 - a) * torch.pow((1 - a_m) * output - (1 - a_m)*target, 2)) )
		return loss




class MaskDataset(data.Dataset):#dataset with mask, return (data, mask)

	def __init__(self, data, mask):
		super(MaskDataset, self).__init__()
		self.data = torch.Tensor(data)
		self.mask = torch.Tensor(mask)
		print('Create MaskDataset, size:', data.shape)


	def __getitem__(self, index):
		return self.data[index], self.mask[index]

	def __len__(self):
		return len(self.data)


class MissMaskDataset(data.Dataset):#dataset with mask and ramdom missing,  return (incomplete_data, complete_data, mask)

	def __init__(self, data, mask, missing_rate):

		def set0(data, rate):
			m = np.random.uniform(0, 1, data.shape)
			n_data = np.copy(data)
			n_data[np.where(m < missing_rate)] = 0 
			return n_data 

		super(MissMaskDataset, self).__init__()	
		self.missing_data = torch.Tensor(set0(data, missing_rate))
		self.mask = torch.Tensor(mask)
		self.raw_data =  torch.Tensor(data)

		print('Create MissMaskDataset, size:', data.shape)

	def __getitem__(self, index):
		return self.missing_data[index], self.raw_data[index], self.mask[index]

	def __len__(self):
		return len(self.raw_data)

def draw_curve(loss_list, label, name):
	plt.xlabel('epoch')
	plt.ylabel(label)
	plt.plot(loss_list)
	plt.savefig(name, dpi=500)


def cat_z_m(x, m):
	x = torch.cat((x.view(x.shape[0], x.shape[1], x.shape[2], 1), m.view(m.shape[0], m.shape[1], m.shape[2], 1)), 3).view(x.shape[0], x.shape[1], -1)
	return x 


def num_param(model, comment=''):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters])
    print(comment + ' Trainable Parameters: %.4f million' % (parameters / 1000000))
    print(comment + ' Usage is %.4f' % (parameters * 32 / 8 / 1024 / 1024))


def get_step_loss(model, z, m):
	r = model(torch.Tensor(z), torch.Tensor(m)).cpu().detach().numpy()
	loss = np.sqrt(np.mean(np.square(m * r - m * z)))
	return loss 