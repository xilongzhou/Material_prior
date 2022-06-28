import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19

class TextureDescriptor(nn.Module):

	def __init__(self, device):
		super(TextureDescriptor, self).__init__()
		self.device = device
		self.outputs = []

		# get VGG19 feature network in evaluation mode
		self.net = vgg19(True).features.to(device)
		self.net.eval()

		# change max pooling to average pooling
		for i, x in enumerate(self.net):
			if isinstance(x, nn.MaxPool2d):
				self.net[i] = nn.AvgPool2d(kernel_size=2)

		def hook(module, input, output):
			self.outputs.append(output)

		#for i in [6, 13, 26, 39]: # with BN
		for i in [4, 9, 18, 27]: # without BN
			self.net[i].register_forward_hook(hook)

		# weight proportional to num. of feature channels [Aittala 2016]
		self.weights = [1, 2, 4, 8, 8]

		# this appears to be standard for the ImageNet models in torchvision.models;
		# takes image input in [0,1] and transforms to roughly zero mean and unit stddev
		self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
		self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

	def forward(self, x):
		self.outputs = []

		# run VGG features
		x = self.net(x)
		self.outputs.append(x)

		result = []
		batch = self.outputs[0].shape[0]

		for i in range(batch):
			temp_result = []
			for j, F in enumerate(self.outputs):

				# print(j, ' shape: ', F.shape)

				F_slice = F[i,:,:,:]
				f, s1, s2 = F_slice.shape
				s = s1 * s2
				F_slice = F_slice.view((f, s))

				# Gram matrix
				G = torch.mm(F_slice, F_slice.t()) / s
				temp_result.append(G.flatten())
			temp_result = torch.cat(temp_result)

			result.append(temp_result)
		return torch.stack(result)

	def eval_CHW_tensor(self, x):
		"only takes a pytorch tensor of size B * C * H * W"
		assert len(x.shape) == 4, "input Tensor cannot be reduced to a 3D tensor"
		x = (x - self.mean) / self.std
		return self.forward(x.to(self.device))


#####################################################################################
################## Texture Descriptor 2 loss with gt and input ######################
#####################################################################################

class TDLoss(nn.Module):
	def __init__(self, GT_img, device, num_pyramid):
		super(TDLoss, self).__init__()
		# create texture descriptor
		self.net_td = TextureDescriptor(device) 
		# fix parameters for evaluation 
		for param in self.net_td.parameters(): 
		    param.requires_grad = False 

		self.num_pyramid = num_pyramid

		self.GT_td = self.compute_td_pyramid(GT_img.to(device))


	def forward(self, img):

		# td1 = self.compute_td_pyramid(img1)
		td = self.compute_td_pyramid(img)

		tdloss = (td - self.GT_td).abs().mean() 

		return tdloss


	def compute_td_pyramid(self, img):
	    """compute texture descriptor pyramid

	    Args:
	        img (tensor): 4D tensor of image (NCHW)
	        num_pyramid (int): pyramid level]

	    Returns:
	        Tensor: 2-d tensor of texture descriptor
	    """    
	    td = self.net_td.eval_CHW_tensor(img) 
	    for scale in range(self.num_pyramid):
	        td_ = self.net_td.eval_CHW_tensor(nn.functional.interpolate(img, scale_factor = 1.0/(2.0**(scale+1)), mode='bilinear', align_corners=True))
	        td = torch.cat([td, td_], dim=1) 
	    return td


#####################################################################################
########## Texture Descriptor loss to evalute between gt and input ##################
#####################################################################################

class TDLoss_2(nn.Module):
	def __init__(self, device, num_pyramid):
		super(TDLoss_2, self).__init__()
		# create texture descriptor
		self.net_td = TextureDescriptor(device) 
		# fix parameters for evaluation 
		for param in self.net_td.parameters(): 
		    param.requires_grad = False 

		self.num_pyramid = num_pyramid


	def forward(self, img, GT):

		GT = self.compute_td_pyramid(GT)
		td = self.compute_td_pyramid(img)

		tdloss = (td - GT).abs().mean() 

		return tdloss


	def compute_td_pyramid(self, img):
	    """compute texture descriptor pyramid

	    Args:
	        img (tensor): 4D tensor of image (NCHW)
	        num_pyramid (int): pyramid level]

	    Returns:
	        Tensor: 2-d tensor of texture descriptor
	    """    
	    td = self.net_td.eval_CHW_tensor(img) 
	    for scale in range(self.num_pyramid):
	        td_ = self.net_td.eval_CHW_tensor(nn.functional.interpolate(img, scale_factor = 1.0/(2.0**(scale+1)), mode='bilinear', align_corners=True))
	        td = torch.cat([td, td_], dim=1) 
	    return td

