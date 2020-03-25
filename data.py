import torch
import torch.utils.data as data
import numpy as np


def TDMinMaxScaler(data):
	#mi = data[np.where(data != np.nan)].min(axis=1).min(axis=0)
	#ma = data[np.where(data != np.nan)].max(axis=1).max(axis=0)
	mi = data.min(axis=2)
	ma = data.max(axis=2)
	#print(mi.shape)



	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			data[i][j] = (data[i][j] - mi[i][j]) / (ma[i][j] - mi[i][j] + 1e-8)
	#data = (data - mi) / (ma - mi)
	return data, ma, mi

class MissDataset(data.Dataset):# return (complete_data, missing_data, mask)

	def __init__(self, data , mask):
		super(MissDataset, self).__init__()

		self.data = torch.Tensor(data)

		self.mask = torch.Tensor(mask)
		print('Create Dataset, size:', data.shape)
		print('Sample, Var, Time')

		

	def __getitem__(self, index):
		return self.data[index],  self.mask[index]

	def __len__(self):
		return len(self.data)


def MissDataLoader(file_path):
	data = np.load(file_path)
	#print('Data loading, shape :', data.shape)
	data_m = np.ones(data.shape)
	data_m[np.where(np.isnan(data))] = 0

	data_x = data[:]
	data_x[np.where(np.isnan(data))] = 0

	data_x, ma, mi = TDMinMaxScaler(data_x)

	return data_x, data_m, ma, mi

