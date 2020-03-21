import torch
import torch.nn as nn
import torch.nn.functional as F 
EPS = 1e-8

class TCDAE(nn.Module):
	def __init__(self, N, B=128, H=128, P=3, X=4, R=2, C=0.2):
		super(TCDAE, self).__init__()
		print(B, H, P, X, R, C)

		#B = N
		self.C = C
		
		self.encoder = Encoder(N, B)
		self.impute_net = TemporalConvNet(B, H, P, X, R)
		self.decoder = Decoder(N, B)

		self.tcnn = nn.Sequential(
			self.encoder,
			self.impute_net,
			self.decoder)

	def forward(self, x, m=None):
		#x  M * N * T 
		x = nn.Dropout(self.C)(x)


		return self.tcnn(x)#M * N * T



class Encoder(nn.Module):
	def __init__(self, N, B):
		super(Encoder, self).__init__()

		self.encode_net = nn.Sequential(
			nn.Conv1d(N, B, 15, padding=7),
			nn.ReLU())


	def forward(self, x):
		v = self.encode_net(x)
		return v

class Decoder(nn.Module):
	def __init__(self, N, B):
		super(Decoder, self).__init__()

		self.decode_net = nn.Sequential(
			nn.Conv1d(B, N, 1),
			nn.Sigmoid())

	def forward(self, x):
		return self.decode_net(x)



class TemporalConvNet(nn.Module):
	def __init__(self, B, H, P, X, R, norm_type="gLN", causal=False,
				 mask_nonlinear='relu'):
		"""

			mask_nonlinear: use which non-linear function to generate mask
		"""
		super(TemporalConvNet, self).__init__()
		# Hyper-parameter
		self.mask_nonlinear = mask_nonlinear
		# Components
		# [M, B, T] -> [M, B, T]
		layer_norm = ChannelwiseLayerNorm(B)

		repeats = []
		for r in range(R):
			blocks = []
			for x in range(X):
				dilation = 2**x
				padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
				blocks += [TemporalBlock(B, H, P, stride=1,
										 padding=padding,
										 dilation=dilation,
										 norm_type=norm_type,
										 causal=causal)]
			repeats += [nn.Sequential(*blocks)]
		temporal_conv_net = nn.Sequential(*repeats)
		#self.repeats = repeats
		# [M, B, K] -> [M, N, K]
		
		# Put together
		self.network = nn.Sequential(layer_norm,
									 #bottleneck_conv1x1,
									 temporal_conv_net)
									# nn.PReLU(),
									# mask_conv1x1)
									 #nn.Sigmoid())

	   # mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)
	   # self.mask_net = nn.Sequential(nn.PReLU(),
									  #  mask_conv1x1,
									  #  nn.Sigmoid())

	def forward(self, x):
		"""
		x: M * B * T
		output M * B * T
		"""
		_, output  = self.network(x)  
		return output
 

class TemporalBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size,
				 stride, padding, dilation, norm_type="gLN", causal=False):
		super(TemporalBlock, self).__init__()
		# [M, B, K] -> [M, H, K]
		conv1x1 = nn.Conv1d(in_channels, out_channels, 3, padding=1,  bias=False)
		prelu = nn.PReLU()
		norm = chose_norm(norm_type, out_channels)
		# [M, H, K] -> [M, B, K]
		dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
										stride, padding, dilation, norm_type,
										causal)
		# Put together
		self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

		#self.att_net = nn.Sequential(nn.Conv1d(in_channels, 1, 3, padding=1),
		#                           nn.PReLU()) # 1 * K 
	   # self.att_net = nn.Sequential(nn.Conv1d(in_channels, in_channels, 3, padding=1),
		  #  nn.Sigmoid())

	def forward(self, x):
		"""
		Args:
			x: [M, B, K], sc:[M, Sc, K] we simply set Sc == B   
		Returns:
			[M, B, K]
		"""
		if type(x) != tuple:
			sc = torch.zeros(x.shape, dtype=torch.float32)
			if x.is_cuda:
				sc = sc.cuda() 
		else:
			sc = x[1]
			x = x[0]

		residual = x
		#out = self.net(x)
		out, new_sc = self.net(x)

		#a = self.att_net(out) # [M, B, K]
		#a = torch.mean(a, dim=2, keepdim=True)# [M, B, 1]

		return out + residual, sc + new_sc# * a   # look like w/o F.relu is better than w/ F.relu
		# return F.relu(out + residual)


class DepthwiseSeparableConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size,
				 stride, padding, dilation, norm_type="gLN", causal=False):
		super(DepthwiseSeparableConv, self).__init__()
		# Use `groups` option to implement depthwise convolution
		# [M, H, K] -> [M, H, K]
		depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
								   stride=stride, padding=padding,
								   dilation=dilation, groups=in_channels,
								   bias=False)
		if causal:
			chomp = Chomp1d(padding)
		prelu = nn.PReLU()
		norm = chose_norm(norm_type, in_channels)
		# [M, H, K] -> [M, B, K]
		#output = nn.Conv1d(in_channels, out_channels, 1, bias=False)
		self.output_conv = nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False)

		self.sc_conv = nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False)

		# Put together
		if causal:
			self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm)#, output)
		else:
			self.net = nn.Sequential(depthwise_conv, prelu, norm)#, output)

	def forward(self, x):
		"""
		Args:
			x: [M, H, K]
		Returns:
			result: [M, B, K]
		"""
		x = self.net(x)
		output = self.output_conv(x)
		sc = self.sc_conv(x)


		return output, sc

def chose_norm(norm_type, channel_size):
	"""The input of normlization will be (M, C, K), where M is batch size,
	   C is channel size and K is sequence length.
	"""
	if norm_type == "gLN":
		return GlobalLayerNorm(channel_size)
	elif norm_type == "cLN":
		return ChannelwiseLayerNorm(channel_size)
	else: # norm_type == "BN":
		# Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
		# along M and K, so this BN usage is right.
		return nn.BatchNorm1d(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
	"""Channel-wise Layer Normalization (cLN)"""
	def __init__(self, channel_size):
		super(ChannelwiseLayerNorm, self).__init__()
		self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
		self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
		self.reset_parameters()

	def reset_parameters(self):
		self.gamma.data.fill_(1)
		self.beta.data.zero_()

	def forward(self, y):
		"""
		Args:
			y: [M, N, K], M is batch size, N is channel size, K is length
		Returns:
			cLN_y: [M, N, K]
		"""
		mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
		var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
		cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
		return cLN_y


class GlobalLayerNorm(nn.Module):
	"""Global Layer Normalization (gLN)"""
	def __init__(self, channel_size):
		super(GlobalLayerNorm, self).__init__()
		self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
		self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
		self.reset_parameters()

	def reset_parameters(self):
		self.gamma.data.fill_(1)
		self.beta.data.zero_()

	def forward(self, y):
		"""
		Args:
			y: [M, N, K], M is batch size, N is channel size, K is length
		Returns:
			gLN_y: [M, N, K]
		"""
		# TODO: in torch 1.0, torch.mean() support dim list
		mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
		var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
		gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
		return gLN_y


