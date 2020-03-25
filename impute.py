import argparse
import os 
import torch
from data import MissDataset, MissDataLoader
from TCDAE import TCDAE
from utils import *

parser = argparse.ArgumentParser('Impute missing data Using TCDAE')

parser.add_argument('--model_path', type=str, 
                    default='./output/final.pth', #required=True,
                    help='Path to model file')

parser.add_argument('--dataset_path', type=str, 
                    default='./input/air.npy' ,
                    help='File path of dataset to impute')

parser.add_argument('--out_dir', type=str, 
                    default='./result',
                    help='Directory putting imputed dataset')

parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU to impute data')


def impute(args, model, z, m):

	tmp_x = model(torch.FloatTensor(z).cuda()) if args.use_cuda else model(torch.FloatTensor(z)) 
	if args.use_cuda:
		tmp_x = tmp_x.cpu()

	tmp_x = tmp_x.detach().numpy()

	x = z * m + tmp_x * (1 - m)
	return x

def re_scale(data, ma, mi):
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			data[i][j] = data[i][j] * (ma[i][j] - mi[i][j] + 1e-8) + mi[i][j]
	return data

def save_data(args, data):
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	file_name = os.path.split(args.dataset_path)[-1].split('.')[0] + '_recover.npy'

	save_path = os.path.join(args.out_dir, file_name)

	np.save(save_path, data)

if __name__ == '__main__':
    args = parser.parse_args()

    data_z, data_m, ma, mi = MissDataLoader(args.dataset_path)
    model = torch.load(args.model_path, map_location=lambda storage, loc: storage)

    model.eval()

    if args.use_cuda:
        model.cuda()

    data_r = impute(args, model, data_z, data_m)


    imputed_data = re_scale(data_r, ma, mi)

    save_data(args, imputed_data)
