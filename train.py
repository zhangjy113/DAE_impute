import argparse
import os 
import torch
from data import MissDataset, MissDataLoader
from TCDAE import TCDAE
from utils import *

import random
import math
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(
    "Time series missing data imputation"
    "with Temporal Convolutional Denoising Autoencoder")

parser.add_argument('--train_path', type=str, default='./input/air.npy',
                    help='directory of train data, with npy format')


parser.add_argument('--B', default=128, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=128, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3 , type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--L', default=2, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=2, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=0.2, type=float,
                    help='Probability of dropout in input layer')

parser.add_argument('--epochs', default=200, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--l2', default=1e-3, type=float,
                    help='weight decay (L2 penalty)')

# minibatch
parser.add_argument('--shuffle', default=1, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size')
#parser.add_argument('--num_workers', default=4, type=int,
#                    help='Number of workers to generate minibatch')

parser.add_argument('--save_folder', default='./output/',
                    help='Location to save epoch models')


def train(dataLoader, args, train_z, train_m):


	model = TCDAE(N=args.N, B=args.B, H=args.H, P=args.P, X=args.L, R=args.R, C=args.C)
	num_param(model)
	model.cuda()
	model  = torch.nn.DataParallel(model) 


	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

	loss_fn = MaskLoss()

	loss_list = []
	best_rmse = 1e8

	for epoch in range(args.epochs):

		model.train()

		for i, (z, m) in enumerate(dataLoader):
			z = Variable(z).cuda()
			m = Variable(m).cuda()

			decoded = model(z, m)

			loss = loss_fn(decoded, z, m)#Denosing: calculate loss by decode and complete data 
			#loss = loss_fn(decoded, raw_x)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


		model.eval()

		step_loss = get_step_loss(model, train_z, train_m)

		
		loss_list.append(step_loss )

		#test_rmse = get_rmse(model, train_x, train_z, train_m)
		print (epoch, 'Step loss:', step_loss)#, '  Test rmse:', test_rmse)
		#if best_rmse > test_rmse:
		#	best_rmse = test_rmse
			
		#	save(model, train_x, train_z, train_m)


	#draw_curve(loss_list, 'RMSE', 'RMSE')
	#print('Best Test RMSE', best_rmse)
	return model 

def main(args):
	train_z, train_m = MissDataLoader(args.train_path)
	train_set = MissDataset(train_z, train_m)

	args.N = train_z.shape[1]
	train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle)

	model = train(train_loader, args, train_z, train_m)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

