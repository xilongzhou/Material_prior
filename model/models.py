import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from collections import OrderedDict
import re
import warnings



def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		n = m.in_channels
		for k in m.kernel_size:
			n*=k
		stdv = 1./np.sqrt(n)
		# print('stdv: ', stdv)
		m.weight.data.normal_(0.0, 0.02)
		# m.weight.data.uniform_(-0.02, 0.02)
		# torch.nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
		if m.bias is not None:
			# m.bias.data.uniform_(-0.02, 0.02)
			m.bias.data.normal_(0,0.02)
			# m.bias.data.fill_(0)

	elif classname.find('Linear') != -1:
		n = m.in_features
		y = np.sqrt(1/float(n))
		# print('input features: ',n)
		m.weight.data.normal_(0.0, y)
		if m.bias is not None:
			m.weight.data.normal_(0.0, y)   
			m.bias.data.normal_(0.0, 0.02) 



class MyConv2d(nn.Conv2d):
    __doc__ = nn.Conv2d.__doc__

    def forward(self, input, params=None):
        if params is None:
            raise ('params should not be None inside my conv 2d !!!!')

        # for p in params:
        # 	print(p, ' ', params[p].shape)
        bias = params.get('bias', None)

        return F.conv2d(input, params['weight'], bias, self.stride,self.padding, groups = self.groups)


class MyNet(nn.Module):
	def __init__(self, opt, device, use_bias=True, sfaB = 1.0, selected_in = 5, nc_in=None):
		super(MyNet, self).__init__()

		self.STN_use=opt.STN_use
		self.pixconv_n = opt.pixconv_n
		self.ch_conv = opt.add_channel_conv
		self.output_nc = opt.output_nc
		self.ngf = opt.ngf
		self.STN_theta = opt.STN_theta
		self.STN_type=opt.STN_type

		self.opt=opt
		self.selected_in = selected_in
		self._children_modules_parameters_cache = {}

		if self.pixconv_n==1:
			self.ngf = self.output_nc

		# using sample net or not
		self.SampleNet_weight_num = 0
		if opt.Sample_Net=='1':
			self.select_input_nc = selected_in
			self.SampleNet_weight_num = nc_in*self.selected_in
			self.sfaB = sfaB
			self.softmax = nn.Softmax(dim=0)
		elif opt.Sample_Net=='2':
			self.select_input_nc = selected_in
			self.SampleNet_weight_num = opt.input_nc*self.selected_in
			setattr(self, 'net_sample', MyConv2d(opt.input_nc, self.select_input_nc, 1, 1, 0, bias=False))
		# no sapmle net
		else:
			self.select_input_nc = opt.input_nc


		## layers for 1 x 1 x m x n conv
		if opt.regconv:

			for i in range(self.pixconv_n):
				## first layer
				if i==0:
					setattr(self, 'net_regconv'+str(i+1), nn.Conv2d(self.select_input_nc, self.ngf, opt.kernel_size, 1, padding_mode='circular',padding=int((opt.kernel_size-1)*0.5), bias=use_bias))

				## last layer
				elif i==self.pixconv_n-1:				
					setattr(self, 'net_regconv'+str(i+1), nn.Conv2d(self.ngf, self.output_nc, opt.kernel_size, 1, padding_mode='circular',padding=int((opt.kernel_size-1)*0.5), bias=use_bias))

				## intermediate layer
				else:				
					setattr(self, 'net_regconv'+str(i+1), nn.Conv2d(self.ngf, self.ngf, opt.kernel_size, 1, padding_mode='circular',padding=int((opt.kernel_size-1)*0.5), bias=use_bias))


		else:
			for i in range(self.pixconv_n):
				## first layer
				if i==0:
					if opt.net_params:
						setattr(self, 'net_pixconv'+str(i+1), MyConv2d(self.select_input_nc, self.ngf, 1, 1, 0, bias=use_bias))
					else:
						setattr(self, 'net_pixconv'+str(i+1), nn.Conv2d(self.select_input_nc, self.ngf, 1, 1, 0, bias=use_bias))

				## last layer
				elif i==self.pixconv_n-1:
					if opt.net_params:
						setattr(self, 'net_pixconv'+str(i+1), MyConv2d(self.ngf, self.output_nc, 1, 1, 0, bias=use_bias))
					else:				
						setattr(self, 'net_pixconv'+str(i+1), nn.Conv2d(self.ngf, self.output_nc, 1, 1, 0, bias=use_bias))

				## intermediate layer
				else:
					if opt.net_params:
						setattr(self, 'net_pixconv'+str(i+1), MyConv2d(self.ngf, self.ngf, 1, 1, 0, bias=use_bias))
					else:				
						setattr(self, 'net_pixconv'+str(i+1), nn.Conv2d(self.ngf, self.ngf, 1, 1, 0, bias=use_bias))

			if self.ch_conv:
				for i in range(self.pixconv_n-1):
					if opt.net_params:
						setattr(self, 'net_chconv'+str(i+1), MyConv2d(self.ngf, self.ngf, opt.kernel_size, 1, padding_mode='circular',padding=int((opt.kernel_size-1)*0.5), bias=use_bias, groups = self.ngf))
					else:				
						setattr(self, 'net_chconv'+str(i+1), nn.Conv2d(self.ngf, self.ngf, opt.kernel_size, 1, padding_mode='circular',padding=int((opt.kernel_size-1)*0.5), bias=use_bias, groups = self.ngf))

		self._count_params()
		print('number of parameters are: ',self._params_len)

		self.relu = nn.ReLU()
		self.leaky_relu = nn.LeakyReLU(0.2)
		self.tan = nn.Tanh()
		self.Sigmoid = nn.Sigmoid()

		self.batch_flag = opt.batch_size>1 and opt.Train_Encoder

		## define the learning network to estimate parameters
		if opt.net_params:
			## encoder --> MLP
			self.MLP_num = opt.MLP_nc if not opt.pnet_enco else self.opt.ngf2*8*2*2
			pnet_out_nc = self.opt.ngf2*8 if not opt.VAE else self.opt.ngf2*16
			if opt.pnet_enco:
				Encoder = nn.Sequential(
					nn.Conv2d(3, self.opt.ngf2, 4, 2, 1, bias=use_bias), # 512 -> 256
					nn.LeakyReLU(0.2),
					nn.Conv2d(self.opt.ngf2, self.opt.ngf2*2, 4, 2, 1, bias=use_bias), # 256 -> 128
					nn.LeakyReLU(0.2),
					nn.Conv2d(self.opt.ngf2*2, self.opt.ngf2*4, 4, 2, 1, bias=use_bias), # 128 -> 64
					nn.LeakyReLU(0.2),
					nn.Conv2d(self.opt.ngf2*4, self.opt.ngf2*8, 4, 2, 1, bias=use_bias), # 64 -> 32
					nn.LeakyReLU(0.2),
					nn.Conv2d(self.opt.ngf2*8, self.opt.ngf2*8, 4, 2, 1, bias=use_bias), # 32 -> 16
					nn.LeakyReLU(0.2),
					nn.Conv2d(self.opt.ngf2*8, self.opt.ngf2*8, 4, 2, 1, bias=use_bias), # 16 -> 8
					nn.LeakyReLU(0.2),
					nn.Conv2d(self.opt.ngf2*8, self.opt.ngf2*8, 4, 2, 1, bias=use_bias), # 8 -> 4
					nn.LeakyReLU(0.2),
					nn.Conv2d(self.opt.ngf2*8, pnet_out_nc, 4, 2, 1, bias=use_bias), # 4 -> 2
					nn.LeakyReLU(0.2),
					# nn.Conv2d(self.opt.ngf2*8, self.opt.ngf2*8, 4, 2, 1, bias=use_bias), # 2 -> 1
					# nn.LeakyReLU(0.2),			
					)
				setattr(self, 'LearnNet_Enc', Encoder)

			self._num_STN = 0
			self._num_STN2 = 0
			if opt.STN_use!='NA':
				if opt.STN_theta=='s1':
					self._num_STN = 1
					temp_shape = 1
					self._STN_shape = torch.Size([1,temp_shape])

				elif opt.STN_theta=='s2':
					self._num_STN = 2
					temp_shape = 2
					self._STN_shape = torch.Size([1,temp_shape])

				if opt.STN_type=='sep':
					if opt.order =='pcon_STN':
						self._num_STN = self._num_STN*self.ngf
						self._STN_shape = torch.Size([self.ngf,temp_shape])

					elif opt.order =='STN_pcon':
						self._num_STN = self._num_STN*self.opt.input_nc	
						self._STN_shape = torch.Size([self.opt.input_nc,temp_shape])

				if opt.add_STN_last:
					if opt.STN_theta=='s1':
						self._num_STN2 = 1
						self._STN2_shape = torch.Size([1,1])
					elif opt.STN_theta=='s2':
						self._num_STN2 = 2
						self._STN2_shape = torch.Size([1,2])

			# self._latent = torch.rand(1, self._MLP_num)#.cuda()
			self.params_total = self._params_len + self._num_STN + self._num_STN2 + self.SampleNet_weight_num

			Enco_MLP=[]
			for mlp in range(opt.n_mlp-1):
				Enco_MLP.append(nn.Linear(self.MLP_num, self.MLP_num, bias=True))
				Enco_MLP.append(nn.LeakyReLU(0.2))
			Enco_MLP.append(nn.Linear(self.MLP_num, self.params_total, bias=True))
			setattr(self, 'LearnNet_MLP', nn.Sequential(*Enco_MLP))

		if opt.Sample_Net:
			self.SampleNet_weight = torch.nn.Parameter(torch.ones((nc_in,selected_in)))
			# self.SampleNet_weight_num = opt.input_nc*self.selected_in
			# self.MLP_num2 = self.opt.ngf2*8*2*2

			# SampleNet=[]
			# SampleNet = nn.Sequential(
			# 	nn.Conv2d(3, self.opt.ngf2, 4, 2, 1, bias=use_bias), # 512 -> 256
			# 	nn.LeakyReLU(0.2),
			# 	nn.Conv2d(self.opt.ngf2, self.opt.ngf2*2, 4, 2, 1, bias=use_bias), # 256 -> 128
			# 	nn.LeakyReLU(0.2),
			# 	nn.Conv2d(self.opt.ngf2*2, self.opt.ngf2*4, 4, 2, 1, bias=use_bias), # 128 -> 64
			# 	nn.LeakyReLU(0.2),
			# 	nn.Conv2d(self.opt.ngf2*4, self.opt.ngf2*8, 4, 2, 1, bias=use_bias), # 64 -> 32
			# 	nn.LeakyReLU(0.2),
			# 	nn.Conv2d(self.opt.ngf2*8, self.opt.ngf2*8, 4, 2, 1, bias=use_bias), # 32 -> 16
			# 	nn.LeakyReLU(0.2),
			# 	nn.Conv2d(self.opt.ngf2*8, self.opt.ngf2*8, 4, 2, 1, bias=use_bias), # 16 -> 8
			# 	nn.LeakyReLU(0.2),
			# 	nn.Conv2d(self.opt.ngf2*8, self.opt.ngf2*8, 4, 2, 1, bias=use_bias), # 8 -> 4
			# 	nn.LeakyReLU(0.2),
			# 	nn.Conv2d(self.opt.ngf2*8, self.opt.ngf2*8, 4, 2, 1, bias=use_bias), # 4 -> 2
			# 	)

			# SampleNet_MLP = nn.Sequential(
			# 	nn.Linear(self.MLP_num2, self.MLP_num2, bias=use_bias),
			# 	nn.LeakyReLU(0.2),
			# 	nn.Linear(self.MLP_num2, self.SampleNet_weight_num, bias=use_bias)
			# 	)

			# setattr(self, 'SampleNet', SampleNet)
			# setattr(self, 'SampleNet_MLP', SampleNet_MLP)

	def _count_params(self):
		self._params_len = 0
		for name, params in self.named_parameters():
			if 'net_' in name:
				self._params_len += params.numel()

	def _STN_func_expl(self,theta,x,step, flag=False):

		all_flag = (self.STN_type=='all') or flag

		if self.opt.STN_pad=='circular':
			_,_,w,h = x.shape
			x = x.tile((1,1,3,3))

		## input scaling theta: [opt.ngf,2]
		if not all_flag:

			# [opt.ngf, 2] --> [opt.ngf, 2, 3]
			theta_temp = torch.zeros(theta.shape[0],6).to(theta.device)

			if self.opt.STN_theta=='s2':
				theta_temp[:,0] = theta[:,0]
				theta_temp[:,4] = theta[:,1]
			if self.opt.STN_theta=='s1':
				theta_temp[:,0] = theta[:,0]
				theta_temp[:,4] = theta[:,0]

			in_theta = theta_temp.view(-1,2,3)

			x=x.permute(1,0,2,3)

			grid = F.affine_grid(in_theta,x.size())
			if self.opt.STN_pad=='circular':
				x = F.grid_sample(x, grid)
			else:
				x = F.grid_sample(x, grid, padding_mode = self.opt.STN_pad)

			x=x.permute(1,0,2,3)

			if step%100==0:
				print('step %d theta shape:'%step, in_theta.shape,'value: ',in_theta[0,...])

		## input scaling theta: [1,2]
		else:
			# [1, 2] --> [1, 2, 3]
			theta_temp = torch.zeros(theta.shape[0],6).to(theta.device)
			if self.opt.STN_theta=='s2':
				theta_temp[:,0] = theta[:,0]
				theta_temp[:,4] = theta[:,1]
			if self.opt.STN_theta=='s1':
				theta_temp[:,0] = theta[:,0]
				theta_temp[:,4] = theta[:,0]

			in_theta = theta_temp.view(-1,2,3)

			grid = F.affine_grid(in_theta, x.size())
			if self.opt.STN_pad=='circular':
				x = F.grid_sample(x, grid)
			else:
				x = F.grid_sample(x, grid, padding_mode = self.opt.STN_pad)

			if step%100==0:
				print('step %d theta shape:'%step, in_theta.shape,'value: ',in_theta)

		if self.opt.STN_pad=='circular':
			x = x[:,:,w:2*w,h:2*h]

		return x			

	def _reparameterize(self,in_tensor, smooth):

		"""
		:param mu: mean from the encoder's latent space
		:param log_var: log variance from the encoder's latent space
		"""
		_,l = in_tensor.shape
		self.mu = in_tensor[:,0:int(l*0.5)]
		self.log_var = in_tensor[:,int(l*0.5):]
		std = torch.exp(0.5*self.log_var) # standard deviation
		eps = torch.randn_like(std) # `randn_like` as we need the same size

		if smooth:
			self.sample  = self.sample + eps*0.2 
			# print('smooth:', self.sample[0,0:5])

		else:
			self.sample = self.mu + (eps * std) # sampling as if coming from the input space
			# print('std',std[0,0:5])
			# print('not smooth:', self.sample[0,0:5])

		return self.sample

	def _add_smooth(self, in_tensor, smooth):
		eps = torch.randn_like(in_tensor) # `randn_like` as we need the same size
		in_tensor = in_tensor + eps*0.2
		return in_tensor

	def _Net_Params(self, input2, noise, smooth, load_latent = None):

		if load_latent is not None:
			params = self.LearnNet_MLP(load_latent)
			return params, load_latent
		else:
			if self.opt.pnet_enco:
				latent = self.LearnNet_Enc(input2)
				flat_latent = torch.flatten(latent,start_dim = 1)
				if self.opt.VAE:
					flat_latent = self._reparameterize(flat_latent, smooth)
				if smooth:
					flat_latent = self._add_smooth(flat_latent, smooth)
				params = self.LearnNet_MLP(flat_latent)
			else:
				flat_latent = torch.flatten(input2,start_dim = 1)	
				# print(flat_latent.shape)
				# print(flat_latent.shape)
				params = self.LearnNet_MLP(flat_latent)
			return params,flat_latent			

	def _Params_to_Weight(self, params, batch_number=0):
		# print(self.LearnNet_MLP[-1].weight.grad[])
		start_index=0
		for name, param in self.named_parameters():

			if 'net_' in name:
				leng = len(torch.flatten(param))
				self._children_modules_parameters_cache[name] = torch.reshape(params[batch_number,start_index:start_index+leng],param.shape)

				# print(name,' start ',start_index,' end ', start_index+leng, 'length: ', leng)
				start_index += leng

		## adding params to STN
		if self.opt.STN_use!='NA':
			self._children_modules_parameters_cache['STN1'] = torch.reshape(params[batch_number,start_index:start_index+self._num_STN],self._STN_shape)
			start_index += self._num_STN
			if self.opt.add_STN_last:
				self._children_modules_parameters_cache['STN2'] = torch.reshape(params[batch_number,start_index:start_index+self._num_STN2], self._STN2_shape)
				start_index += self._num_STN2	

		if self.opt.Sample_Net=='1':
			self.SampleNet_weight = nn.Sigmoid()(torch.reshape(params[batch_number,start_index:],(self.opt.input_nc,self.selected_in)))

	def get_subdict(self, in_key):

		subdict={}

		if in_key+'.weight' in self._children_modules_parameters_cache:
			subdict['weight'] = self._children_modules_parameters_cache[in_key+'.weight']

		if in_key+'.bias' in self._children_modules_parameters_cache:
			subdict['bias'] = self._children_modules_parameters_cache[in_key+'.bias']

		return subdict

	def forward(self, input_pat, step, theta, theta2, device, noise=None, input2=None, load_latent=None, smooth=False):

		########### estimation network or not ################
		params=None
		latent=None
		if self.opt.net_params:
			if load_latent is not None:
				params,latent = self._Net_Params(input2, noise, smooth, load_latent = load_latent)
			else:
				params,latent = self._Net_Params(input2, noise, smooth)

			if noise is not None:
				noise = torch.reshape(noise,params.shape)
				params = params + noise

		############## main network ####################
		## if batchsize > 1, run the network in parallel (only for train encoder)
		if self.batch_flag:

			if self.opt.Sample_Net=='1':
				_,c,w,h = input_pat.shape
				temp_weight = self.sfaB * step**2
				reshape_in = input_pat.permute(0,2,3,1).view(-1,c)

				# self.SampleNet_weight = torch.flatten(self.SampleNet(input2),start_dim = 1)
				# self.SampleNet_weight = nn.Sigmoid()(torch.reshape(self.SampleNet_MLP(self.SampleNet_weight),(self.opt.batch_size,self.opt.input_nc,self.selected_in)))

			final_layer=torch.empty(0).to(device)
			for batch in range(self.opt.batch_size):

				self._Params_to_Weight(params, batch_number = batch)

				if self.opt.Sample_Net=='1':
					matrix_weight = self.softmax(temp_weight*self.SampleNet_weight).to(device)
					if step%self.opt.save_freq==0:
						print('matrix_weight: ',matrix_weight)
						print('self.SampleNet_weight: ',self.SampleNet_weight)
					Sample_net_out = torch.matmul(reshape_in, matrix_weight)
					selected_input = Sample_net_out.view(w,h,-1).permute(2,0,1).unsqueeze(0)

				elif self.opt.Sample_Net=='2':
					selected_input = nn.Sigmoid()(self.net_sample(input_pat, params = self.get_subdict('net_sample')))

				else:
					selected_input = input_pat

				# print(selected_input.shape)
				layer=[]
				for j in range(self.pixconv_n):
					pixconv = getattr(self, 'net_pixconv'+str(j+1))

					# first layer
					if j==0:

						if self.opt.net_params:
							layer.append(self.leaky_relu(pixconv(selected_input, params = self.get_subdict('net_pixconv'+str(j+1)))))
						else:
							layer.append(self.leaky_relu(pixconv(selected_input)))

						# print('1st layer ',layer[-1].shape)

						if self.ch_conv:
							chconv = getattr(self, 'net_chconv'+str(j+1))
							if self.opt.net_params:
								layer.append(self.leaky_relu(chconv(layer[-1], params = self.get_subdict('net_chconv'+str(j+1)))))
							else:
								layer.append(self.leaky_relu(chconv(layer[-1])))

					# last layer
					elif j==self.pixconv_n-1:

						if self.opt.net_params:
							layer.append(self.leaky_relu(pixconv(layer[-1], params = self.get_subdict('net_pixconv'+str(j+1)))))
						else:
							layer.append(self.leaky_relu(pixconv(layer[-1])))

					# other layer
					else:
						if self.opt.net_params:
							layer.append(self.leaky_relu(pixconv(layer[-1], params = self.get_subdict('net_pixconv'+str(j+1)))))
						else:
							layer.append(self.leaky_relu(pixconv(layer[-1])))

						if self.ch_conv:
							chconv = getattr(self, 'net_chconv'+str(j+1))
							if self.opt.net_params:
								layer.append(self.leaky_relu(chconv(layer[-1], params = self.get_subdict('net_chconv'+str(j+1)))))
							else:
								layer.append(self.leaky_relu(chconv(layer[-1])))

						# if self.STN:
						# 	layer.append(self.STN_func(layer[-1], j, step))

				final_layer = torch.cat((final_layer,layer[-1]),dim=0)

			return [final_layer], theta, theta2, latent, (selected_input+1)*0.5

		## otherwise no, for optimization
		else:
			if self.opt.net_params:
				self._Params_to_Weight(params)

			## if Sample Net
			if self.opt.Sample_Net=='1':
				_,c,w,h = input_pat.shape
				# temp_weight = self.sfaB * step**2 if step > 500 else self.sfaB * step * 0.5
				temp_weight = self.sfaB * step
				reshape_in = input_pat.permute(0,2,3,1).view(-1,c)

				# self.SampleNet_weight = torch.flatten(self.SampleNet(input2),start_dim = 1)
				# self.SampleNet_weight = nn.Sigmoid()(torch.reshape(self.SampleNet_MLP(self.SampleNet_weight),(self.opt.input_nc,self.selected_in)))

				matrix_weight = self.softmax(temp_weight * self.SampleNet_weight).to(device)
				if step%500==0:
					print('matrix_weight: ',matrix_weight)
					
				Sample_net_out = torch.matmul(reshape_in, matrix_weight)
				selected_input = Sample_net_out.view(w,h,-1).permute(2,0,1).unsqueeze(0)

			elif self.opt.Sample_Net=='2':
				selected_input = nn.Sigmoid()(self.net_sample(input_pat, params = self.get_subdict('net_sample')))

			else:
				selected_input = input_pat

			layer=[]
			for j in range(self.pixconv_n):

				if self.opt.regconv:
					regconv = getattr(self, 'net_regconv'+str(j+1))

					# first layer
					if j==0:
						layer.append(self.leaky_relu(regconv(selected_input)))
					# last layer
					elif j==self.pixconv_n-1:
						layer.append(self.leaky_relu(regconv(layer[-1])))
					# other layer
					else:
						layer.append(self.leaky_relu(regconv(layer[-1])))

				else:
					pixconv = getattr(self, 'net_pixconv'+str(j+1))

					# first layer
					if j==0:
						if self.opt.net_params:
							layer.append(self.leaky_relu(pixconv(selected_input, params = self.get_subdict('net_pixconv'+str(j+1)))))
						else:
							layer.append(self.leaky_relu(pixconv(selected_input)))
						if self.ch_conv:
							chconv = getattr(self, 'net_chconv'+str(j+1))
							if self.opt.net_params:
								layer.append(self.leaky_relu(chconv(layer[-1], params = self.get_subdict('net_chconv'+str(j+1)))))
							else:
								layer.append(self.leaky_relu(chconv(layer[-1])))

					# last layer
					elif j==self.pixconv_n-1:
						if self.opt.net_params:
							layer.append(self.Sigmoid(pixconv(layer[-1], params = self.get_subdict('net_pixconv'+str(j+1)))))
						else:
							layer.append(self.Sigmoid(pixconv(layer[-1])))

					# other layer
					else:
						if self.opt.net_params:
							layer.append(self.leaky_relu(pixconv(layer[-1], params = self.get_subdict('net_pixconv'+str(j+1)))))
						else:
							layer.append(self.leaky_relu(pixconv(layer[-1])))
						if self.ch_conv:
							chconv = getattr(self, 'net_chconv'+str(j+1))
							if self.opt.net_params:
								layer.append(self.leaky_relu(chconv(layer[-1], params = self.get_subdict('net_chconv'+str(j+1)))))
							else:
								layer.append(self.leaky_relu(chconv(layer[-1])))


			return layer, theta, theta2, latent, (selected_input+1)*0.5

