import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
np.random.seed(0)
torch.manual_seed(0)
# Inspo https://github.com/leoniloris/1D-Convolutional-Variational-Autoencoder/blob/master/CNN_VAE.ipynb


# (De)convolution block

class Conv_block(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding, is_conv=True):
		super(Conv_block, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.padding = padding 
		self.pool_op = torch.nn.AvgPool1d(2, ) if is_conv \
				  else torch.nn.Upsample(scale_factor=2, mode='linear')
		self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
		self.bn = torch.nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.99)
		self.relu = torch.nn.ReLU()
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return self.pool_op(x)

# Encoder block

class Encoder(torch.nn.Module):
	def __init__(self, in_channels, in_length, nclasses, latent_size, encoder_out_channels, device):
		super(Encoder, self).__init__()
		self.device = device

		self.in_channels = in_channels
		self.in_length = in_length
		self.nclasses = nclasses
		self.latent_size = latent_size
		self.encoder_out_channels = encoder_out_channels
		length = self.in_length
		self.bn0 = torch.nn.BatchNorm1d(self.in_channels, eps=0.001, momentum=0.99)
		# Layer 1
		in_channels = self.in_channels
		out_channels = 32
		kernel_size = 101#201
		padding = kernel_size // 2
		self.conv_block_1 = Conv_block(in_channels, out_channels, kernel_size, padding)
		length = length // 2
		# Layer 2
		in_channels = out_channels
		out_channels = 32
		kernel_size = 101#101201
		padding = kernel_size // 2
		self.conv_block_2 = Conv_block(in_channels, out_channels, kernel_size, padding)
		length = length // 2
		
		# Layer 3
		in_channels = out_channels
		last_featuremaps_channels = 64
		kernel_size = 101#201
		padding = kernel_size // 2
		self.conv_block_3 = Conv_block(in_channels, last_featuremaps_channels, kernel_size, padding)
		length = length // 2
		
		in_channels = last_featuremaps_channels
		out_channels = nclasses
		kernel_size = 15#30
		padding = kernel_size // 2
		self.conv_final = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
		self.gp_final = torch.nn.AvgPool1d(length)
		
		# encoder
		in_channels = last_featuremaps_channels
		out_channels = self.encoder_out_channels
		kernel_size = 25#51
		padding = kernel_size // 2
		self.adapt_pool = torch.nn.AvgPool1d(2); length = length // 2
		self.adapt_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
		self.encode_mean = torch.nn.Linear(length*out_channels, self.latent_size)
		self.encode_logvar = torch.nn.Linear(length*out_channels, self.latent_size)
		self.relu = torch.nn.ReLU()
		length = 1

	def forward(self, x):
		x = x.view(-1, self.in_channels, self.in_length)
		x = self.bn0(x)
		x = self.conv_block_1(x)
		#print('x1', x.shape)
		x = self.conv_block_2(x)
		#print('x2', x.shape)
		x = self.conv_block_3(x)
		#print('x3', x.shape)

		cv_final = self.conv_final(x)
		#print('x4', x.shape)

		oh_class = self.gp_final(cv_final)
		x = self.adapt_pool(x)
		#print('x5', x.shape)

		x = self.adapt_conv(x)
		#print('x6', x.shape)

		x = x.view(x.size(0), -1)
		mean = self.relu(self.encode_mean(x)) 
		logvar = self.relu(self.encode_logvar(x))
		#return oh_class.view(oh_class.size(0), self.nclasses), mean, 
		return self._sample_latent(mean, logvar), mean, logvar#addition#logvar, 
		
	def _sample_latent(self, mean, logvar): # z ~ N(mean, var (sigma^2))  
		#print('mean', mean)
		#print('logvar', logvar) 
		np.random.seed(0)
		z_std = torch.from_numpy(np.random.normal(0, 1, size=mean.size())).float()
		sigma = torch.exp(logvar).to(self.device)
		return mean + sigma * Variable(z_std, requires_grad=False).to(self.device)#, z_std


# Decoder block

class Decoder(torch.nn.Module):
	def __init__(self, length, in_channels, nclasses, latent_size, device):
		super(Decoder, self).__init__()
		self.device=device
		self.in_channels = in_channels
		self.length = length
		self.latent_size = latent_size
		length = self.length  
		print('l1',length)
		length = length // 2 // 2 // 2 
		print('l2', length)
		# Adapt Layer
		self.relu = torch.nn.ReLU()
		self.tanh = torch.nn.Tanh()
		self.adapt_nn = torch.nn.Linear(latent_size, self.in_channels*length)
		# Layer 1
		in_channels = self.in_channels
		out_channels = 64
		kernel_size = 101#201
		padding = kernel_size // 2 #100#kernel_size // 2 + (length - 1) * 2 % 2 - 1#kernel_size // 2 + (length % 2) - 1
		#print('padding', padding)
		self.deconv_block_1 = Conv_block(in_channels, out_channels, kernel_size, padding, is_conv=False)
		length = length * 2
		#print('l3', length)
		# Layer 2
		in_channels = out_channels
		out_channels = 32
		kernel_size = 101#201
		padding = kernel_size // 2
		self.deconv_block_2 = Conv_block(in_channels, out_channels, kernel_size, padding, is_conv=False)
		length = length * 2
		#print('l4', length)
 
		# Layer 3
		in_channels = out_channels
		out_channels = 32
		kernel_size = 101#201
		padding = kernel_size // 2 
		self.deconv_block_3 = Conv_block(in_channels, out_channels, kernel_size, padding, is_conv=False)
		length = length * 2
		#print('l4', length)

		in_channels = out_channels
		out_channels = 1
		kernel_size = 101#201
		padding = kernel_size // 2
		self.decode_conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
		
	def forward(self, z):

		x = self.relu(self.adapt_nn(z)).to(self.device)
		#print('s1', x.shape)
		x = x.view(x.size(0), self.in_channels, self.length // 2 // 2 // 2)
		#print('s2', x.shape)
		x = self.deconv_block_1(x)
		#print('s3', x.shape)
		x = self.deconv_block_2(x)
	   # print('s4', x.shape)
		x = self.deconv_block_3(x)
		#print('s5', x.shape)
		x = self.decode_conv(x)
		#print('s6', x.shape)
		out = self.tanh(x)
		return out


# VAE

class VAE_1D(torch.nn.Module):
	def __init__(self, length, nclasses, latent_size, transition_channels, device):
		super(VAE_1D, self).__init__()
		self.encoder = Encoder(1, length, nclasses, latent_size, transition_channels, device)
		self.decoder = Decoder(length, transition_channels, nclasses, latent_size, device)
	def count_parameters(self):
		return np.sum([np.prod(x.size()) for x in self.parameters()])
	def forward(self, x):
		#oh_class, mean, z = self.encoder(x)
		z, mean, logvar = self.encoder(x)
		x_decoded = self.decoder(z)
		return z, x_decoded #oh_class, mean, z, x_decoded