
import numpy as np
import os
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from skimage.transform import resize

from torch import autograd

eps=1e-6

# convert to float tensor
to_tensor = lambda a: torch.as_tensor(a, dtype=torch.float)

# Convert [-1, 1] to [0, 1]
to_zero_one = lambda a: a / 2.0 + 0.5



def set_params(opt, device):

	size = 4.0

	light_pos = torch.tensor([0.0, 0.0, 4], dtype=torch.float32).view(1, 3, 1, 1)
	light = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).view(1, 3, 1, 1) * 16 * np.pi

	light_pos = light_pos.to(device)
	light = light.to(device)

	return light, light_pos, size


def AdotB(a, b):
	return (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)


def norm(vec): #[B,C,W,H]
	vec = vec.div(vec.norm(2.0, 1, keepdim=True))
	return vec

def roll_row(img_in, n):
	return img_in.roll(-n, 2)


def roll_col(img_in, n):
	return img_in.roll(-n, 3)

# Check if the input tensor is a 1 channel tensor    
def grayscale_input_check(tensor_input, err_var_name):
	assert tensor_input.shape[1] == 1, '%s should be a grayscale image' % err_var_name


def mkdirs(paths):
	if isinstance(paths, list) and not isinstance(paths, str):
		for path in paths:
			mkdir(path)
	else:
		mkdir(paths)

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

## read images
def read_image(filename):
	img = imageio.imread(filename)
	if img.dtype == np.float32:
		return img
	if img.dtype == np.uint8:
		return img.astype(np.float32) / 255.0
	elif img.dtype == np.uint16:
		return img.astype(np.float32) / 65535.0

def Debug_net_params(net, out='Shape'):
	for name,param in net.named_parameters():
		if out=='Shape':
			print(name, 'shape: ',param.shape)
		elif out=='Value':
			print(name, 'value: ', param.squeeze().squeeze().squeeze())


## visulaize intermediate layer
def vis_interlayer_gif(opt, step,out, gif_c, gif_g, Resize=True):

	for num_layer in range(len(out[:-1])):

		layer_viz = out[num_layer][0, :, :, :].detach().cpu().numpy()

		min_val = np.amin(layer_viz)
		max_val = np.amax(layer_viz)

		## gray scale feature maps
		fig1, axis = plt.subplots(int(opt.ngf/4), 4, figsize=(4,4))
		fig1.suptitle('step %d'%step, fontsize=10)
		axis = axis.flatten()
		for n_c, ax in enumerate(axis):
			if n_c<layer_viz.shape[0]:
				# ax.imshow(layer_viz[n_c,:,:], cmap='gray')#,vmin=0, vmax=1)
				ax.imshow(layer_viz[n_c,:,:], cmap='gray',vmin=min_val, vmax=max_val)
				ax.axis("off")
				ax.set_aspect('equal')
			else:
				continue

		fig1.subplots_adjust(wspace=0, hspace=0.02)

		# color feature maps
		# fig2, axis2 = plt.subplots(int(opt.ngf/4), 4, figsize=(4,4))
		# fig2.suptitle('step %d'%step, fontsize=10)
		# axis2 = axis2.flatten()
		# for n_c2, ax2 in enumerate(axis2):
		# 	if n_c2<layer_viz.shape[0]:
		# 	color_img = vis_interlayer(layer_viz[n_c2,:,:])
		# 	ax2.imshow(color_img)#,vmin=0, vmax=1)
		# 	ax2.axis("off")
		# 	ax2.set_aspect('equal')
		# fig2.subplots_adjust(wspace=0, hspace=0.02)

		npy1 = fig2data(fig1)
		# npy2 = fig2data(fig2)

		# if resize==True:

		gif_g[num_layer].append((npy1*255).astype(np.uint8))
		# gif_c[num_layer].append((npy2*255).astype(np.uint8))

		plt.close(fig1)
		# plt.close(fig2)
		
		# imageio.imwrite('%dtest2.png'%num_layer, (npy2*255).astype(np.uint8))
		# imageio.imwrite('%dtest.png'%num_layer, (npy1*255).astype(np.uint8))

## save intermediate layers to gif
def save_interlayer_gif(opt, gif_c, gif_g, gif_layer_path):

	# assert len(gif_c) == len(gif_g), "the length of gif color and gray should be same"
	# for c_i in range(len(gif_c)):
	# 	saved_path = gif_layer_path+'color%d.gif'%c_i
	# 	imageio.mimsave(saved_path, gif_c[c_i], format='GIF', fps=opt.FPS, loop=2)
	# 	optimize(saved_path)

	for g_i in range(len(gif_g)):
		saved_path = gif_layer_path+'gray%d.gif'%g_i
		imageio.mimsave(saved_path, gif_g[g_i],format='GIF', fps=opt.FPS, loop=2)
		optimize(saved_path)


## save images to gif
def save_images_gif(opt, gif_img, gif_img_path, fps = None, savename=None):
	if fps is None:
		myfps =  opt.FPS
	else:
		myfps = fps

	if isinstance(gif_img, dict):
		for key in gif_img:
			if savename is not None:
				saved_path = gif_img_path+'%s_%s.gif'%(savename, key)
			else:
				saved_path = gif_img_path+'%s.gif'%(key)
			imageio.mimsave(saved_path, gif_img[key], format='GIF', fps=myfps, loop=2)
			optimize(saved_path)
	else:
		saved_path = gif_img_path+'ren.gif' 
		imageio.mimsave(saved_path, gif_img, format='GIF', fps=myfps, loop=2)
		optimize(saved_path)		

## matplot fig to numpy
def fig2data(fig):
	import PIL.Image as Image
	from io import BytesIO

	buffer_ = BytesIO()
	fig.savefig(buffer_,format='png')
	# buffer_.seek(0)

	fig.canvas.draw()

	w,h = fig.get_size_inches() * fig.get_dpi()
	w,h = int(w), int(h)

	buf = np.fromstring(fig.canvas.tostring_argb(),dtype=np.uint8)
	buf.shape = (w,h,4)

	buf = np.roll(buf,3,axis=2)

	image=Image.frombytes('RGBA', (w,h),buf.tostring())

	image=np.asarray(image)/255.

	return image[:,:,:3]


## save loss function
def save_loss(loss_dict, save_dir, step, save_name=None):

	if save_name is None:

		plt.figure()
		for i in loss_dict:
			# print(loss_dict[i])
			plt.plot(step, loss_dict[i], label='%s' % i)
		plt.legend()
		plt.savefig(save_dir+'/losses.png')
		plt.close()

		plt.figure()
		for i in loss_dict:
			# print('log: ', np.log1p(loss_dict[i]))
			plt.plot(step, np.log1p(loss_dict[i]), label='%s' % i)
		plt.legend()
		plt.savefig(save_dir+'/losses_log.png')
		plt.close()

	else:

		plt.figure()
		for i in loss_dict:
			# print(loss_dict[i])
			plt.plot(step, loss_dict[i], label='%s' % i)
		plt.legend()
		plt.savefig(save_dir+'/losses%s.png'%save_name)
		plt.close()

		plt.figure()
		for i in loss_dict:
			# print('log: ', np.log1p(loss_dict[i]))
			plt.plot(step, np.log1p(loss_dict[i]), label='%s' % i)
		plt.legend()
		plt.savefig(save_dir+'/losses%s_log.png'%save_name)
		plt.close()

## save loss function
def scale_vs_loss(loss_dict, save_dir, scale):

	plt.figure()
	for i in loss_dict:
		# print(loss_dict[i])
		plt.plot(scale, loss_dict[i], label='%s' % i)
	plt.legend()
	plt.savefig(save_dir+'/scale_vs_loss.png')
	plt.close()



## save generated output maps and rendering
def save_output_dict(opt, dicts, step, base_dir, gif_images=None, epoch=None, saveBatch=False):

	for name, img in dicts.items():

		## normalize height map for visualization
		if name=='height':
			max_val = torch.max(img)
			min_val = torch.min(img)
			img = (img - min_val)/(max_val - min_val)

		img = img.detach().cpu().numpy()
		# img = np.tile(img, (2,2))

		if 'in_pat' not in name or 'selected_in_pat' in name:
			if opt.Train_Encoder:
				filename =  base_dir + "iter%d_"%(epoch) + name if epoch is not None else base_dir + "iter%d_"%(epoch) + name
			else:
				filename =  base_dir + name + "_iter%d"%(epoch) if epoch is not None else base_dir + name + "_iter%d"%step
		else:
			filename = base_dir + name
				# print('inpat')

		if len(img.shape) == 3:
			out_img = np.transpose(img, [1, 2, 0])
			out_img_saved = (out_img * 255.0).astype(np.uint8)

		elif len(img.shape)==4:
			if not saveBatch:
				out_img = np.transpose(img[0,...], [1, 2, 0])
				out_img_saved = (out_img * 255.0).astype(np.uint8)
			else:
				out_img_1 = np.transpose(img[0,...], [1, 2, 0])
				out_img_2 = np.transpose(img[1,...], [1, 2, 0])
				out_img_3 = np.transpose(img[2,...], [1, 2, 0])
				out_img_4 = np.transpose(img[3,...], [1, 2, 0])

				out_img_saved1 = (out_img_1 * 255.0).astype(np.uint8)
				out_img_saved2 = (out_img_2 * 255.0).astype(np.uint8)
				out_img_saved3 = (out_img_3 * 255.0).astype(np.uint8)
				out_img_saved4 = (out_img_4 * 255.0).astype(np.uint8)


		if gif_images:
			fig = plt.figure(figsize=(4,4))
			plt.suptitle('%s step %d'%(name,step))
			if name=='height' or name=='out_maps':
				plt.imshow(out_img, cmap='gray')
			else:
				plt.imshow(out_img)
			plt.axis('off')
			out_npy = fig2data(fig)
			gif_images[name].append((out_npy * 255.0).astype(np.uint8))

			plt.close(fig)
		else:
			# print("Saving output ...")
			if 'in_pat' in name:
			# if False:
				n = out_img_saved.shape[-1]
				# print(n)
				for num_out in range(n):
					imageio.imwrite(filename+'_%d.png'%num_out, out_img_saved[:,:,num_out:num_out+1])
			else:
				if not saveBatch:
					imageio.imwrite(filename+'.png', out_img_saved)
				else:
					imageio.imwrite(filename+'_1.png', out_img_saved1)
					imageio.imwrite(filename+'_2.png', out_img_saved2)
					imageio.imwrite(filename+'_3.png', out_img_saved3)
					imageio.imwrite(filename+'_4.png', out_img_saved4)


## tex [B,C,W,H] to maps ready for rendering
def tex2map(opt, tex, device, inten=3.0):

	## debug Spatial Transformer Net (with single output)
	if opt.debug_STN:
		if not opt.only_STN:
			return nn.Sigmoid()(tex)
		else:
			return (tex+1)*0.5

	if opt.debug_render:
		normal = tex[:,0:3,:,:]*2-1
		albedo = tex[:,3:6,:,:]**2.2
		rough = tex[:,6:7,:,:].expand(-1,3,-1,-1)

		specular = torch.tensor([0.04]).expand(tex.shape[0], 3, tex.shape[2], tex.shape[3])

		output = torch.cat((normal, albedo, rough, specular), dim=1)

		return output

	# normal, albedo, rough, spec 
	if tex.shape[1]==9:

		# normal_x  = tex[:,0,:,:].clamp(-1,1)
		# normal_y  = tex[:,1,:,:].clamp(-1,1)
		# normal_xy = (normal_x**2 + normal_y**2).clamp(min=0, max=1-eps)
		# normal_z  = (1 - normal_xy).sqrt()
		# normal    = torch.stack((normal_x, normal_y, normal_z), 1)
		# normal    = normal.div(normal.norm(2.0, 1, keepdim=True))

		normal_x  = tex[:,0,:,:].clamp(0,1)*2-1
		normal_y  = tex[:,1,:,:].clamp(0,1)*2-1
		normal_xy = (normal_x**2 + normal_y**2).clamp(min=0, max=1-eps)
		normal_z  = (1 - 0*normal_xy).sqrt()
		normal    = torch.stack((normal_x, normal_y, normal_z), 1)
		normal    = normal.div(normal.norm(2.0, 1, keepdim=True))

		albedo = tex[:,2:5,:,:].clamp(0,1)**2.2

		rough = tex[:,5,:,:].clamp(0,1)
		rough = rough.unsqueeze(1).expand(-1,3,-1,-1)

		specular = tex[:,6:9,:,:].clamp(0,1)**2.2 

		output = torch.cat((normal, albedo, rough, specular), dim=1)
		return output

	# height, albedo, rough
	elif tex.shape[1]==5:

		# print('height: ',tex[0,0:1,0,0:5])
		# print('albedo: ',tex[0,1:4,0,0:5])
		# print('rough: ',tex[0,4:5,0,0:5])


		## this is for [-1,1] tex
		# height = (tex[:,0:1,:,:].clamp(-1,1)+1)*0.5 # with tanh
		height = tex[:,0:1,:,:] # no tanh
		normal = height_to_normal(height, device, max_intensity=inten)*2-1 #[0,1] --> [-1,1]

		# albedo = (tex[:,1:4,:,:].clamp(-1,1)+1)*0.5
		albedo = tex[:,1:4,:,:] # use sigmoid

		# rough = nn.Sigmoid()(tex[:,4,:,:])*0.9+0.1 # use sigmoid
		rough = tex[:,4,:,:]#.clamp(min=0.05) # use sigmoid
		rough = rough.unsqueeze(1).expand(-1,3,-1,-1)

		## fixed specular
		specular = torch.tensor([0.04]).expand(tex.shape[0], 3, tex.shape[2], tex.shape[3]).to(device)

		output = torch.cat((normal, albedo, rough, specular), dim=1)
		return output,height

	# from image albedo + normal + rough + spec
	elif tex.shape[1]==10:
		normal = tex[:,0:3,:,:]*2-1
		albedo = tex[:,3:6,:,:]**2.2
		rough = tex[:,6:7,:,:].expand(-1,3,-1,-1)
		# rough = torch.ones_like(albedo)
		specular = tex[:,7:10,:,:]**2.2

		output = torch.cat((normal, albedo, rough, specular), dim=1)

		return output

	# from image albedo + normal + rough
	elif tex.shape[1]==7:
		tex = (tex+1)*0.5
		normal = tex[:,0:3,:,:]*2-1
		albedo = tex[:,3:6,:,:]**2.2
		rough = tex[:,6:7,:,:].expand(-1,3,-1,-1)

		specular = torch.tensor([0.04]).expand(tex.shape[0], 3, tex.shape[2], tex.shape[3]).to(device)

		output = torch.cat((normal, albedo, rough, specular), dim=1)

		return output, tex[:,6:7,:,:]


## maps to png, ready for logging and saving
def map2png(maps, isSpecular=False):
	# maps = maps.detach().cpu().numpy()
	normal = (maps[:,0:3,:,:]+1)*0.5
	albedo = maps[:,3:6,:,:]**(1/2.2)
	# test = (maps[:,3:6,:,:] <= 0)
	# print('<0: ',test.any())
	# print('NAN: ',torch.isnan(albedo).any())
	rough = maps[:,6:9,:,:]
	if isSpecular:
		metal = maps[:,9:12,:,:]**(1/2.2)
	else:
		metal = maps[:,9:12,:,:]

	return normal, albedo, rough, metal



def height_to_normal(img_in, device, mode='tangent_space', normal_format='gl', use_input_alpha=False, use_alpha=False, intensity=1.0/3.0, max_intensity=3.0):
	"""Atomic function: Normal (https://docs.substance3d.com/sddoc/normal-172825289.html)

	Args:
		img_in (tensor): Input image.
		mode (str, optional): 'tangent space' or 'object space'. Defaults to 'tangent_space'.
		normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
		use_input_alpha (bool, optional): Use input alpha. Defaults to False.
		use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
		intensity (float, optional): Normalized height map multiplier on dx, dy. Defaults to 1.0/3.0.
		max_intensity (float, optional): Maximum height map multiplier. Defaults to 3.0.

	Returns:
		Tensor: Normal image.
	"""
	grayscale_input_check(img_in, "input height field")

	img_size = img_in.shape[2]
	# intensity = intensity * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
	intensity = (intensity * 2.0 - 1.0) * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
	dx = roll_col(img_in, -1) - img_in
	dy = roll_row(img_in, -1) - img_in
	if normal_format == 'gl':
		img_out = torch.cat((intensity*dx, -intensity*dy, torch.ones_like(dx)), 1)
	elif normal_format == 'dx':
		img_out = torch.cat((intensity*dx, intensity*dy, torch.ones_like(dx)), 1)
	else:
		img_out = torch.cat((-intensity*dx, intensity*dy, torch.ones_like(dx)), 1)
	img_out = norm(img_out)
	if mode == 'tangent_space':
		img_out = img_out / 2.0 + 0.5
	
	if use_alpha == True:
		if use_input_alpha:
			img_out = torch.cat([img_out, img_in], dim=1)
		else:
			img_out = torch.cat([img_out, torch.ones(img_out.shape[0], 1, img_out.shape[2], img_out.shape[3])], dim=1)

	return img_out




## height to normal world unit
def height_to_normal_world_units(img_in, normal_format='gl', sampling_mode='standard', use_alpha=False, surface_size=0.3, max_surface_size=1000.0,height_depth=0.16, max_height_depth=100.0):
	"""Non-atomic function: Height to Normal World Units (https://docs.substance3d.com/sddoc/height-to-normal-world-units-159450573.html)

	Args:
		img_in (tensor): Input image.
		normal_format (str, optional): 'gl' or 'dx'. Defaults to 'gl'.
		sampling_mode (str, optional): 'standard' or 'sobel', switches between two sampling modes determining accuracy. Defaults to 'standard'.
		use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
		surface_size (float, optional): Normalized dimensions of the input Heightmap. Defaults to 0.3.
		max_surface_size (float, optional): Maximum dimensions of the input Heightmap (cm). Defaults to 1000.0.
		height_depth (float, optional): Normalized depth of heightmap details. Defaults to 0.16.
		max_height_depth (float, optional): Maximum depth of heightmap details. Defaults to 100.0.

	Returns:
		Tensor: Normal image.
	"""
	# Check input validity
	grayscale_input_check(img_in, 'input image')
	assert normal_format in ('dx', 'gl'), "normal format must be 'dx' or 'gl'"
	assert sampling_mode in ('standard', 'sobel'), "sampling mode must be 'standard' or 'sobel'"

	surface_size = to_tensor(surface_size) * max_surface_size
	height_depth = to_tensor(height_depth) * max_height_depth

	# Standard normal conversion
	if sampling_mode == 'standard':
		img_out = height_to_normal(img_in, mode = "tangent_space", normal_format=normal_format, use_alpha=use_alpha, intensity=to_zero_one(height_depth / surface_size), max_intensity=256.0)
		# img_out = normal(img_in, mode = "object_space", normal_format=normal_format, use_alpha=use_alpha, max_intensity=256.0)
 
	return img_out


# input [W,H], output [W,H,3]
def vis_interlayer(img):

	img_pos = np.where(img > 0, img, 0)
	img_neg = np.where(img < 0, -img, 0)

	# print('pos num:', np.count_nonzero(img_pos==0), ' neg num:', np.count_nonzero(img_neg==0))

	color_img = np.expand_dims(np.zeros_like(img),axis=-1)
	color_img = np.repeat(color_img,[3], axis=-1)

	# r
	img_pos_min=np.amin(img_pos)
	img_pos_max=np.amax(img_pos)
	if img_pos_min == img_pos_max:
		color_img[:,:,0] = img_pos
	else:
		color_img[:,:,0] = (img_pos - img_pos_min)/(img_pos_max-img_pos_min)

	# b
	img_neg_min=np.amin(img_neg)
	img_neg_max=np.amax(img_neg)
	if img_neg_min == img_neg_max:
		color_img[:,:,2] = img_neg
	else:
		color_img[:,:,2] = (img_neg - img_neg_min)/(img_neg_max-img_neg_min)

	# print('img_pos: ',img_pos)
	# print('img_neg: ',img_neg)
	# print('color: ',color_img)

	return color_img


### compute the gradient penalty of discriminator
### https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)),dtype=torch.float32, device = device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # print(interpolates.dtype)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False, device=device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # gradients = gradients.view(gradients.size(0), -1)
    # print(gradients.shape)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
