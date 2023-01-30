import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

from os import listdir
from os.path import join
import copy

import torch
import random
from model.models import *
from option.base_option import BaseOptions
from utils.util import *
from utils.render import render, getTexPos, affine_img
from utils.descriptor import TDLoss, TDLoss_2
from utils.dataloader import *
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from skimage.transform import resize
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader

from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

def freeze_net(in_net, reset):
	for name, params in in_net.named_parameters():
		if reset:
			params.requires_grad=True
		else:
			if ('localization' not in name) and ('fc_loc' not in name):
				params.requires_grad=False
				print(name)

def load_input_rand(opt):

	# path
	img_format = ['.png', '.jpg', '.jpeg']
	common_pat_path = './data/Patterns2/rand_pat/common'
	class_pat_path = opt.in_pat_path
	in_img_path = opt.in_img_path		
	Gaussian_noise=False

	###################### load patterns ######################

	# load from common pool
	concat_in_pat = torch.empty(0)
	allcommon_pat = os.listdir(common_pat_path)
	common_pat_namelist = []
	while len(common_pat_namelist) < opt.N_common:
		select_pat = allcommon_pat[random.randint(0,len(allcommon_pat)-1)]
		select_pat_name = select_pat.split('.')[0][:-2]
		# print('select pat name: ', select_pat_name)
		if common_pat_namelist:
			ADD=True
			for temp_item in common_pat_namelist:
				if select_pat_name in temp_item:
					ADD=False
					continue
			if ADD:
				common_pat_namelist.append(select_pat)
		else:
			common_pat_namelist.append(select_pat)

	for i in common_pat_namelist:
		ext = os.path.splitext(i)[1]
		if ext not in img_format:
			continue
		## resize image if needed
		temp_in = read_image(join(common_pat_path,i))
		if temp_in.shape[0] != opt.res:
			temp_in = torch.from_numpy(resize(temp_in, (opt.res, opt.res), anti_aliasing=True))
		else:		
			temp_in = torch.from_numpy(temp_in)
		# gray scale image
		if len(temp_in.shape)==2:
			temp_in = temp_in.unsqueeze(-1)
		else:
			temp_in = temp_in[:,:,:1]

		if Gaussian_noise:
			temp_in = torch.rand_like(temp_in)

		concat_in_pat = torch.cat((concat_in_pat,temp_in),dim=-1)

	# load from class pool
	allclass_pat = os.listdir(class_pat_path)
	if opt.myclass!='tiles':
		class_pat_namelist = []
		while len(class_pat_namelist) < opt.N_class:
			select_pat = allclass_pat[random.randint(0,len(allclass_pat)-1)]
			select_pat_name = select_pat.split('.')[0][:-2]
			# print('select pat name: ', select_pat_name)
			if class_pat_namelist:
				ADD=True
				for temp_item in class_pat_namelist:
					if select_pat_name in temp_item:
						ADD=False
						continue
				if ADD:
					class_pat_namelist.append(select_pat)
			else:
				class_pat_namelist.append(select_pat)
	else:
		class_pat_namelist = []
		if opt.test:
			class_pat_namelist.append(allclass_pat[random.randint(0,len(allclass_pat)-1)])					
		else:
			class_pat_namelist = []
			select_pat_name = opt.in_img.split('.')[0] + '_0.png'
			class_pat_namelist.append(select_pat_name)
	for i in class_pat_namelist:
		ext = os.path.splitext(i)[1]
		if ext not in img_format:
			continue
		## resize image if needed
		temp_in = read_image(join(class_pat_path,i))
		if temp_in.shape[0] != opt.res:
			temp_in = torch.from_numpy(resize(temp_in, (opt.res, opt.res), anti_aliasing=True))
		else:		
			temp_in = torch.from_numpy(temp_in)

		if Gaussian_noise:
			temp_in = torch.rand_like(temp_in)

		# gray scale image
		if len(temp_in.shape)==2:
			temp_in = temp_in.unsqueeze(-1)
		else:
			temp_in = temp_in[:,:,:1]
		concat_in_pat = torch.cat((concat_in_pat,temp_in),dim=-1)

	concat_in_pat = concat_in_pat.permute(2,0,1) #[W,H,C]->[C,W,H]

	################## loading image ##################
	in_img_list = ['%s' % opt.in_img]
	concat_in_img = torch.empty(0)
	for j in in_img_list:

		ext = os.path.splitext(j)[1]
		if ext not in img_format:
			continue

		## resize image if needed
		temp_img = read_image(join(in_img_path,j))
		if temp_img.shape[0] != opt.res:
			temp_in = torch.from_numpy(resize(temp_img, (opt.res,opt.res), anti_aliasing=True))
		else:		
			temp_in = torch.from_numpy(temp_img)

		# gray scale image
		if len(temp_in.shape)==2:
			temp_in = temp_in.unsqueeze(-1)
		elif len(temp_in.shape)==3 and temp_in.shape[-1]==4:
			temp_in = temp_in[:,:,:3]

		concat_in_img = torch.cat((concat_in_img,temp_in),dim=-1)
	
	concat_in_img = concat_in_img.permute(2,0,1) #[W,H,C]->[C,W,H]

	return concat_in_pat, concat_in_img, common_pat_namelist+class_pat_namelist


def load_custom(opt):

	# load imag
	temp_img = read_image(join(in_img_path,opt.in_img))
	if temp_img.shape[0] != opt.res:
		temp_in = torch.from_numpy(resize(temp_img, (opt.res,opt.res), anti_aliasing=True))
	else:		
		temp_in = torch.from_numpy(temp_img)
	image_in = temp_in.permute(2,0,1) #[W,H,C]->[C,W,H]
	print(f"loading image done {opt.in_img} ......")

	# load patterns
	concat_in_pat = torch.empty(0)
	for pat in opt.in_pat_path:
		ext = os.path.splitext(i)[1]
		if ext not in img_format:
			continue
		## resize image if needed
		temp_in = read_image(join(opt.in_pat_path,pat))
		if temp_in.shape[0] != opt.res:
			temp_in = torch.from_numpy(resize(temp_in, (opt.res, opt.res), anti_aliasing=True))
		else:		
			temp_in = torch.from_numpy(temp_in)

		# gray scale image
		if len(temp_in.shape)==2:
			temp_in = temp_in.unsqueeze(-1)
		else:
			temp_in = temp_in[:,:,:1]
		concat_in_pat = torch.cat((concat_in_pat,temp_in),dim=-1)

	concat_in_pat = concat_in_pat.permute(2,0,1) #[W,H,C]->[C,W,H]
	print(f"loading patterns done {opt.in_img} ......")
	return concat_in_pat, image_in

def load_edit(opt, test=False):

	# edit_img_path =  './data/Edit'
	edit_img_path = './data/Edit2'

	all_imgs = []
	for in_img in os.listdir(edit_img_path):
		in_img_path = os.path.join(edit_img_path, in_img)

		## resize image if needed
		temp_img = read_image(in_img_path)
		if temp_img.shape[0] != opt.res:
			temp_in = torch.from_numpy(resize(temp_img, (opt.res,opt.res), anti_aliasing=True))
		else:		
			temp_in = torch.from_numpy(temp_img)

		# colro image, remove alpha if necessary
		if len(temp_in.shape)==3 and temp_in.shape[-1]==4:
			temp_in = temp_in[:,:,:3]

		print('in img shape: ', temp_in.shape)
		temp_in = temp_in.permute(2,0,1).unsqueeze(0) #[W,H,C]->[C,W,H]

		all_imgs.append(temp_in**2.2)

	all_imgs = torch.cat(all_imgs, dim=0)

	return all_imgs



def load_input_highres(opt, test=False, tr_pat=None, path=None):

	# path
	print('load high res')
	img_format = ['.png', '.jpg', '.jpeg']
	common_pat_path = path
	in_img_path = opt.in_img_path		

	################################# from common pool
	concat_in_pat = torch.empty(0)

	for i in os.listdir(common_pat_path):
		ext = os.path.splitext(i)[1]
		if ext not in img_format:
			continue
		## resize image if needed
		temp_in = read_image(join(common_pat_path,i))	
		temp_in = torch.from_numpy(temp_in)
		# gray scale image
		if len(temp_in.shape)==2:
			temp_in = temp_in.unsqueeze(-1)
		else:
			temp_in = temp_in[:,:,:1]
		concat_in_pat = torch.cat((concat_in_pat,temp_in),dim=-1)

	concat_in_pat = concat_in_pat.permute(2,0,1) #[W,H,C]->[C,W,H]

	########################################################
	################## loading image list ##################
	########################################################
	in_img_list = ['%s' % opt.in_img]

	concat_in_img = torch.empty(0)
	for j in in_img_list:
		## only load images with png and jpg
		ext = os.path.splitext(j)[1]
		if ext not in img_format:
			continue

		## resize image if needed
		temp_img = read_image(join(in_img_path,j))
		if temp_img.shape[0] != opt.res:
			temp_in = torch.from_numpy(resize(temp_img, (opt.res,opt.res), anti_aliasing=True))
		else:		
			temp_in = torch.from_numpy(temp_img)

		# gray scale image
		if len(temp_in.shape)==2:
			temp_in = temp_in.unsqueeze(-1)
		elif len(temp_in.shape)==3 and temp_in.shape[-1]==4:
			temp_in = temp_in[:,:,:3]

		concat_in_img = torch.cat((concat_in_img,temp_in),dim=-1)
	
	# print('in img shape: ', concat_in_img.shape)
	concat_in_img = concat_in_img.permute(2,0,1) #[W,H,C]->[C,W,H]


	return concat_in_pat, concat_in_img


def optim(opt, net, inpat_data, gt_img, device, init_scale='', loss_list=None, inpat_name_list=None, edit_imgs=None):
 
	light, light_pos, size = set_params(opt, device)
	tex_pos = getTexPos(opt.res, size, device).unsqueeze(0)
	tex_pos_t = getTexPos(opt.res*2, size, device).unsqueeze(0)

	if len(gt_img.shape)==3:
		gt_img = gt_img.unsqueeze(0).to(device)
	else:
		gt_img = gt_img.to(device)

	if opt.edit or opt.test or opt.resume:
		light_opt = torch.load(join(opt.load_ckpt,'ckpt.pt'))['light']
		height_opt = torch.load(join(opt.load_ckpt,'ckpt.pt'))['height']
	else:
		light_opt=torch.tensor([1.0]).cuda()
		height_opt = torch.tensor([opt.H_intensity]).cuda()

	print('height_opt: ', height_opt)
	print('light_opt: ', light_opt)

	if opt.scale_opt and not opt.edit and not opt.test and not opt.resume:
		light_opt = torch.tensor([1.0]).cuda().requires_grad_(True)
		if opt.no_optim_height:
			Optimizer = torch.optim.Adam(list(net.parameters())+[light_opt], lr = opt.lr)
		else:
			height_opt = torch.tensor([opt.H_intensity]).cuda().requires_grad_(True)
			Optimizer = torch.optim.Adam([{'params':list(net.parameters())+[light_opt], 'lr': opt.lr},
										{'params':[height_opt], 'lr': opt.lr}	
										])
	else:
		Optimizer = torch.optim.Adam(list(net.parameters()), lr = opt.lr)

	# decay lr 
	if opt.decay_lr:
		opt_scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=1000, gamma=0.5)

	if opt.net_params:		
		if opt.pnet_enco:
			input_enco = 2*gt_img-1
		else:
			input_enco = torch.rand(1, net.MLP_num).to(device)
	else:
		input_enco = None


	# ## explicitly define STN
	theta_STN2 = None
	theta_STN = None
	# if opt.STN_use=='expl' and not opt.net_params:

	# 	if opt.STN_theta=='s1':
	# 		theta_STN = torch.tensor([[init_scale]], device=device)
	# 		print('scaling theta STN ')
	# 	elif opt.STN_theta=='s2':
	# 		theta_STN = torch.tensor([[init_scale,init_scale]], device=device)
	# 		print('scaling theta STN ')

	# 	if opt.STN_type=='sep':
	# 		if opt.order =='pcon_STN':
	# 			theta_STN = theta_STN.repeat(opt.ngf,1)
	# 		elif opt.order =='STN_pcon':
	# 			theta_STN = theta_STN.repeat(in_pat.shape[1],1)

	# 		print('seperate theta parameters for different STN')
	# 	else:
	# 		print('one common theta parameters for different STN')

	# 	theta_STN=theta_STN.clone().requires_grad_(True)
	# 	print('explicitly setting parameters for theta STN ', theta_STN.shape)

	# 	if opt.add_STN_last:
	# 		if opt.STN_theta=='s1':
	# 			theta_STN2 = torch.tensor([[1.0]], device=device)
	# 			print('scaling theta STN x,y together')
	# 		elif opt.STN_theta=='s2':
	# 			theta_STN2 = torch.tensor([[1.0,1.0]], device=device)
	# 			print('scaling theta STN x,y seperately ')

	# 		theta_STN2=theta_STN2.clone().requires_grad_(True)
	# 		print('explicitly setting parameters for theta STN2 ', theta_STN2.shape)
	# 		Optimizer_theta_STN = torch.optim.Adam([theta_STN]+[theta_STN2], lr = opt.STN_lr)
	# 	else:
	# 		Optimizer_theta_STN = torch.optim.Adam([theta_STN], lr = opt.STN_lr)

	## set up loss type
	criterionTD = TDLoss(gt_img, device, opt.TD_pyramid)

	criterionL1 = torch.nn.L1Loss()
	if opt.loss=='TD':
		print('using texture descriptor loss')
	elif opt.loss=='L1':
		print('using L1 loss')
	elif opt.loss=='TD+L1':
		print('using Texture descriptor loss + L1 loss')

	common_path = join(opt.checkpoints_dir, opt.myclass, opt.name2+'_'+opt.name_pf, str(init_scale))

	## create dirs
	save_path = join(common_path, 'imgs/')
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	print('opt.load_ckpt_option: ', opt.load_ckpt_option)
	print('save_path: ', save_path)

	loss_edit = torch.tensor([0],device=device)
	loss_TD = torch.tensor([0],device=device)
	loss_L1 = torch.tensor([0],device=device)
	step_list = []
	loss_dict = {'TD': [], 'L1': [], 'edit': []}

	# optim
	if not opt.test and not opt.edit and not opt.tile:

		for step in range(opt.total_iter+1):

			in_pats = inpat_data[0:1,...]

			out,theta,theta2,latent,selected_input = net(in_pats*2-1, step, theta_STN, theta_STN2, device, input2=input_enco)

			# print(out[-1].shape)
			out_maps, out_height = tex2map(opt, out[-1], device, inten=height_opt)
			
			out_ren = render(out_maps, tex_pos, light*light_opt, light_pos).clamp(0,1)

			if opt.loss=='TD':
				loss_TD = criterionTD(out_ren)
			elif opt.loss=='TD+L1':
				loss_L1 = opt.lambda_L1*criterionL1(out_ren, gt_img) 
				loss_TD = opt.lambda_TD*criterionTD(out_ren)
			elif opt.loss=='L1':
				loss_L1 = criterionL1(out_ren, gt_img) 
			elif opt.loss=='TD+L1Mean':
				out_mean = out_ren.mean(dim=(-2,-1),keepdim=True)
				gt_mean =  gt_img.mean(dim=(-2,-1),keepdim=True)
				loss_L1 = opt.lambda_L1*criterionL1(out_mean, gt_mean) 
				loss_TD = opt.lambda_TD*criterionTD(out_ren)
			elif opt.loss=='TD+16L1':
				scale_factor = np.log2(opt.res/16)
				out_16 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
				gt_16 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
				loss_L1 = opt.lambda_L1*criterionL1(out_16, gt_16) 
				loss_TD = opt.lambda_TD*criterionTD(out_ren)
			elif opt.loss=='TD+32L1':
				scale_factor = np.log2(opt.res/32)
				out_32 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
				gt_32 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
				loss_L1 = opt.lambda_L1*criterionL1(out_32, gt_32) 
				loss_TD = opt.lambda_TD*criterionTD(out_ren)
			elif opt.loss=='TD+64L1':
				scale_factor = np.log2(opt.res/64)
				out_64 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
				gt_64 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
				loss_L1 = opt.lambda_L1*criterionL1(out_64, gt_64) 
				loss_TD = opt.lambda_TD*criterionTD(out_ren)
			elif opt.loss=='16L1':
				scale_factor = np.log2(opt.res/16)
				out_16 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
				gt_16 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
				loss_L1 = opt.lambda_L1*criterionL1(out_16, gt_16) 

			if opt.edit:
				# out_mean = out_maps.mean(dim=(-2,-1),keepdim=True)
				# loss_edit = criterionL1(out_mean[:,3:6,:,:],color_tensor)*0.1 + criterionL1(out_mean[:,6:9,:,:],rough_tensor)*0.1
				# loss_edit = criterionL1(out_mean[:,6:9,:,:],rough_tensor)*0.1
				loss_edit = criterionTD_edit(out_maps[:,3:6,:,:])*opt.w_edit


			total_loss = loss_TD + loss_L1 + loss_edit

			if step>50:
				loss_dict['TD'].append(loss_TD.item())
				loss_dict['L1'].append(loss_L1.item())
				loss_dict['edit'].append(loss_edit.item())
				step_list.append(step)

			## save output image, and logging loss
			if step%opt.save_freq==0:
				save_loss(loss_dict, common_path, step_list)

				temp_lr = Optimizer.param_groups[0]['lr']
				print('epoch: %d, lr: %f, totalloss: %f, lossL1: %f, lossTD: %f, lossedit: %f'%(step, temp_lr, total_loss, loss_L1, loss_TD, loss_edit))

				N,D,R,S = map2png(out_maps)
				visuals = OrderedDict({
										'normal': N,									
										'height': out_height,
										'albedo': D,
										'rough': R,
										'render': out_ren,
										})

				# tile output
				# if step==opt.total_iter:
				# 	out_t = torch.tile(out[-1], (2,2))
				# 	out_maps_t, _ = tex2map(opt, out_t, device, inten=height_opt)
				# 	out_ren_t = render(out_maps_t, tex_pos_t, light*light_opt, light_pos, device).clamp(0,1)
				# 	N_t,D_t,R_t,S_t = map2png(out_maps_t)
				# 	visuals.update({'normal_t': N_t,									
				# 					'albedo_t': D_t,
				# 					'rough_t': R_t,
				# 					'render_t': out_ren_t,
				# 					})

				if step==0:
				# if True:
					visuals.update({'gt_img':gt_img, 
									'in_pat': in_pats
									})

				if opt.loss=='TD+L1Mean':
					visuals.update({'out_mean':out_mean.expand(1,3,out_ren.shape[-2],out_ren.shape[-1]), 
									'gt_mean': gt_mean.expand(1,3,out_ren.shape[-2],out_ren.shape[-1])
									})				

				if '16L1' in opt.loss:
					visuals.update({'out_16':out_16, 
									'gt_16': gt_16
									})	
				if '32L1' in opt.loss:
					visuals.update({'out_32':out_32, 
									'gt_32': gt_32
									})	
				if '64L1' in opt.loss:
					visuals.update({'out_64':out_64, 
									'gt_64': gt_64
									})	

				if opt.Optim_pat and step!=0:
					visuals.update({
									'in_pat': in_pat
									})

				save_output_dict(opt, visuals, step, save_path)


			Optimizer.zero_grad()
			total_loss.backward()
			Optimizer.step()

			if opt.decay_lr:
				opt_scheduler.step()
		
		del visuals

		# save model
		save_net_path = join(common_path,'ckpt.pt')

		torch.save({'net':net.state_dict(), 'height':height_opt,'light':light_opt}, save_net_path)
		print('height opt: ', height_opt )
		print('light opt: ', light_opt )

	# edit
	if opt.edit:
		in_pats = inpat_data[0:1,...]

		for edit_idx in range(edit_imgs.shape[0]):
			edit_img = edit_imgs[edit_idx:edit_idx+1,...].to(device)
			temp_model = copy.deepcopy(net)
			Optimizer = torch.optim.Adam(list(temp_model.parameters()), lr = opt.lr)

			criterionTD_edit = TDLoss(edit_img, device, opt.TD_pyramid)

			for step in range(opt.total_iter+1):

				out,_,_,_,_ = temp_model(in_pats*2-1, step, theta_STN, theta_STN2, device)

				# print(out[-1].shape)
				out_maps, out_height = tex2map(opt, out[-1], device, inten=height_opt)
				out_ren = render(out_maps, tex_pos, light*light_opt, light_pos).clamp(0,1)

				if opt.loss=='TD':
					loss_TD = criterionTD(out_ren)
				elif opt.loss=='TD+L1':
					loss_L1 = opt.lambda_L1*criterionL1(out_ren, gt_img) 
					loss_TD = opt.lambda_TD*criterionTD(out_ren)
				elif opt.loss=='L1':
					loss_L1 = criterionL1(out_ren, gt_img) 
				elif opt.loss=='TD+L1Mean':
					out_mean = out_ren.mean(dim=(-2,-1),keepdim=True)
					gt_mean =  gt_img.mean(dim=(-2,-1),keepdim=True)
					loss_L1 = opt.lambda_L1*criterionL1(out_mean, gt_mean) 
					loss_TD = opt.lambda_TD*criterionTD(out_ren)
				elif opt.loss=='TD+16L1':
					scale_factor = np.log2(opt.res/16)
					out_16 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					gt_16 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					loss_L1 = opt.lambda_L1*criterionL1(out_16, gt_16) 
					loss_TD = opt.lambda_TD*criterionTD(out_ren)
				elif opt.loss=='TD+32L1':
					scale_factor = np.log2(opt.res/32)
					out_32 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					gt_32 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					loss_L1 = opt.lambda_L1*criterionL1(out_32, gt_32) 
					loss_TD = opt.lambda_TD*criterionTD(out_ren)
				elif opt.loss=='TD+64L1':
					scale_factor = np.log2(opt.res/64)
					out_64 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					gt_64 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					loss_L1 = opt.lambda_L1*criterionL1(out_64, gt_64) 
					loss_TD = opt.lambda_TD*criterionTD(out_ren)
				elif opt.loss=='16L1':
					scale_factor = np.log2(opt.res/16)
					out_16 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					gt_16 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					loss_L1 = opt.lambda_L1*criterionL1(out_16, gt_16) 

				out_mean = out_maps.mean(dim=(-2,-1),keepdim=True)
				# loss_edit = criterionL1(out_mean[:,3:6,:,:],color_tensor)*0.1 + criterionL1(out_mean[:,6:9,:,:],rough_tensor)*0.1
				loss_edit = criterionL1(out_maps[:,6:9,:,:],edit_img)*opt.w_edit
				# loss_edit = criterionTD_edit(out_maps[:,3:6,:,:])*opt.w_edit

				total_loss = loss_TD + loss_L1 + loss_edit

				## save output image, and logging loss
				if step%opt.save_freq==0:
					save_loss(loss_dict, common_path, step_list)

					temp_lr = Optimizer.param_groups[0]['lr']
					print('epoch: %d, lr: %f, totalloss: %f, lossL1: %f, lossTD: %f, lossedit: %f'%(step, temp_lr, total_loss, loss_L1, loss_TD, loss_edit))

					N,D,R,S = map2png(out_maps)
					visuals = OrderedDict({
											'normal_%d'%(edit_idx): N,									
											'height_%d'% (edit_idx): out_height,
											'albedo_%d'% (edit_idx): D,
											'rough_%d'% (edit_idx): R,
											'render_%d'% (edit_idx): out_ren,
											})

					if step==0:
						visuals.update({'gt_img':gt_img, 
										'in_pat': in_pats,
										'edit_%d'% (edit_idx): edit_img**(1/2.2), 
											})					
			

					save_output_dict(opt, visuals, step, save_path)


				Optimizer.zero_grad()
				total_loss.backward()
				Optimizer.step()

			
			del visuals

	# direct test
	if opt.test and inpat_data.shape[0]>1:
		with torch.no_grad():
			for test_idx in range(inpat_data.shape[0]-1):

				in_pats = inpat_data[test_idx+1:test_idx+2,...]
				out,__,__,__,__ = net(in_pats*2-1, 0, theta_STN, theta_STN2, device, input2=input_enco)

				out_maps, out_height = tex2map(opt, out[-1], device, inten=height_opt)
				out_ren = render(out_maps, tex_pos, light*light_opt, light_pos).clamp(0,1)

				## save output image, and logging loss
				N,D,R,S = map2png(out_maps)
				visuals = OrderedDict({'normal%d'%test_idx: N,									
										'height%d'%test_idx: out_height,
										'albedo%d'%test_idx: D,
										'rough%d'%test_idx: R,
										'render%d'%test_idx: out_ren,
										'in_pat%d'%test_idx: in_pats,
										})


				# tile
				# out_t = torch.tile(out[-1], (2,2))
				# out_maps_t, _ = tex2map(opt, out_t, device, inten=height_opt if opt.scale_opt else opt.intensity)
				# out_ren_t = render(out_maps_t, tex_pos, light*light_opt, light_pos, device).clamp(0,1)
				# N_t,D_t,R_t,S_t = map2png(out_maps_t)
				# visuals = update({'normal_t%d'%test_idx: N_t,									
				# 				'albedo_t%d'%test_idx: D_t,
				# 				'rough_t%d'%test_idx: R_t,
				# 				'render_t%d'%test_idx: out_ren_t,
				# 				})

				save_output_dict(opt, visuals, 0, save_path)

	# fine-tune test
	if opt.test and inpat_data.shape[0]>1:
		for test_idx in range(inpat_data.shape[0]-1):

			print(f'......start finetuning {test_idx}th pattern.....')

			# copy the model
			temp_model = copy.deepcopy(net)

			Optimizer = torch.optim.Adam(list(temp_model.parameters()), lr = opt.lr)

			in_pats = inpat_data[test_idx+1:test_idx+2,...]

			loss_TD = torch.tensor([0],device=device)
			loss_L1 = torch.tensor([0],device=device)

			for step in range(opt.total_iter+1):

				out,__,__,__,__ = temp_model(in_pats*2-1, step, theta_STN, theta_STN2, device, input2=input_enco)
				out_maps, out_height = tex2map(opt, out[-1], device, inten=height_opt)
				out_ren = render(out_maps, tex_pos, light*light_opt, light_pos).clamp(0,1)

				if opt.loss=='TD':
					loss_TD = criterionTD(out_ren)
				elif opt.loss=='TD+L1':
					loss_L1 = opt.lambda_L1*criterionL1(out_ren, gt_img) 
					loss_TD = opt.lambda_TD*criterionTD(out_ren)
				elif opt.loss=='L1':
					loss_L1 = criterionL1(out_ren, gt_img) 
				elif opt.loss=='TD+L1Mean':
					out_mean = out_ren.mean(dim=(-2,-1),keepdim=True)
					gt_mean =  gt_img.mean(dim=(-2,-1),keepdim=True)
					loss_L1 = opt.lambda_L1*criterionL1(out_mean, gt_mean) 
					loss_TD = opt.lambda_TD*criterionTD(out_ren)
				elif opt.loss=='TD+16L1':
					scale_factor = np.log2(opt.res/16)
					out_16 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					gt_16 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					loss_L1 = opt.lambda_L1*criterionL1(out_16, gt_16) 
					loss_TD = opt.lambda_TD*criterionTD(out_ren)
				elif opt.loss=='TD+32L1':
					scale_factor = np.log2(opt.res/32)
					out_32 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					gt_32 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					loss_L1 = opt.lambda_L1*criterionL1(out_32, gt_32) 
					loss_TD = opt.lambda_TD*criterionTD(out_ren)
				elif opt.loss=='TD+64L1':
					scale_factor = np.log2(opt.res/64)
					out_64 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					gt_64 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					loss_L1 = opt.lambda_L1*criterionL1(out_64, gt_64) 
					loss_TD = opt.lambda_TD*criterionTD(out_ren)
				elif opt.loss=='16L1':
					scale_factor = np.log2(opt.res/16)
					out_16 = nn.functional.interpolate(out_ren, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					gt_16 = nn.functional.interpolate(gt_img, scale_factor = 1.0/(2.0**scale_factor), mode='bilinear', align_corners=True)
					loss_L1 = opt.lambda_L1*criterionL1(out_16, gt_16) 

				total_loss = loss_TD + loss_L1

				Optimizer.zero_grad()
				total_loss.backward()
				Optimizer.step()

				## save output image, and logging loss
				if step%opt.save_freq==0:

					print('step: %d, totalloss: %f, lossL1: %f, lossTD: %f'%(step, total_loss, loss_L1, loss_TD))




			## save output image, and logging loss
			N,D,R,S = map2png(out_maps)
			visuals = OrderedDict({
									'normal%d_finetune'%test_idx: N,									
									'height%d_finetune'%test_idx: out_height,
									'albedo%d_finetune'%test_idx: D,
									'rough%d_finetune'%test_idx: R,
									'render%d_finetune'%test_idx: out_ren,
									})

			save_output_dict(opt, visuals, step, save_path)


			print(f'.....finish finetuning {test_idx}th pattern......')

	if opt.tile:

		in_pats = inpat_data[0:1,...]

		out,_,_,_,_ = net(in_pats*2-1, 0, theta_STN, theta_STN2, device, input2=input_enco)
		out_maps, out_height = tex2map(opt, out[-1], device, inten=height_opt)
		
		print(out[-1].shape)
		out_t = torch.tile(out[-1], (2,2))
		out_maps_t, _ = tex2map(opt, out_t, device, inten=height_opt)
		out_ren_t = render(out_maps_t, tex_pos_t, light*light_opt, light_pos).clamp(0,1)
		N_t,D_t,R_t,S_t = map2png(out_maps_t)
		visuals = OrderedDict({'normal_t': N_t,									
						'albedo_t': D_t,
						'rough_t': R_t,
						'render_t': out_ren_t,
						})

		save_output_dict(opt, visuals, 0, save_path)

	return 

def filter_opt(opt):

	if opt.Train_Encoder:
		opt.vis_interlayer=False
		opt.aug_traindata = True
		opt.aug_inpats = True

	# load pattern or network if needed
	if opt.edit or opt.resume or opt.test:
		opt.load_ckpt = join(opt.checkpoints_dir, opt.myclass, opt.name2+ '_'+opt.load_pf)

	if opt.load_option=='rand':
		opt.in_img_path = os.path.join(opt.real_root_path, opt.myclass)
		opt.in_pat_path = os.path.join('./data/Patterns2/rand_pat', opt.myclass)
	elif opt.load_option=='cust':
		opt.in_img_path = opt.real_root_path

	if opt.myclass=='tiles':
		opt.N_common = 3
		opt.N_class = 1

if __name__ == "__main__":

	opt = BaseOptions().parse()

	if torch.cuda.is_available() and opt.gpu >= 0:
		torch.cuda.set_device(opt.gpu)
		device = torch.device('cuda')
		print('use GPU')
	else:
		device = torch.device('cpu')
		print('use CPU')

	torch.manual_seed(opt.seed)
	random.seed(opt.seed)
	np.random.seed(opt.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	# filter options
	filter_opt(opt)

	if opt.load_ckpt != '':
		print(opt.load_ckpt)
		# input patterns for optimization
		if opt.load_option=='rand':
			_, in_imgs, _ = load_input_rand(opt)
		elif opt.load_option=='highres':
			in_patterns, in_imgs = load_input_highres(opt, path = './data/Patterns2/highres_pat/highres')
			in_patterns = in_patterns.to(device).unsqueeze(0)

		if 'pat' in opt.load_ckpt_option:
			load_pat = torch.load(join(opt.load_ckpt,'inpat.pt'))
			in_patterns = load_pat.to(device).unsqueeze(0)
			opt.input_nc = in_patterns.shape[1]
			print(opt.load_ckpt, ' loaded pattern successfully!!')

		print('in_patterns: ',in_patterns.shape)

	else:
		# load input
		if opt.load_option=='rand':
			in_patterns_tr, in_imgs, _ = load_input_rand(opt)
		elif opt.load_option=='highres':
			in_patterns_tr, in_imgs = load_input_highres(opt, path = './data/Patterns2/highres_pat/lowres')
		elif opt.load_option=="cust":
			in_patterns_tr, in_imgs = load_custom()

		# path to save pattern
		save_pat_path = join(opt.checkpoints_dir, opt.myclass, opt.name2+'_'+opt.name_pf,'inpat.pt')
		torch.save(in_patterns_tr, save_pat_path)

		# channel number of 1st layer
		opt.input_nc = in_patterns_tr.shape[0]
		in_patterns = in_patterns_tr.unsqueeze(0).to(device) if len(in_patterns_tr.shape)==3 else in_patterns_tr.to(device)

	# for testing, load test patterns
	if opt.test:
		if opt.load_option=='class':
			test_patterns = load_input_class(opt, test=True)
			in_patterns = torch.cat([in_patterns, test_patterns], dim=0)

		elif opt.load_option=='rand':
			test_pat_list=[]
			for i in range(4):
				test_patterns = load_input_rand(opt, test=True, tr_pat = in_patterns.squeeze(0)).unsqueeze(0).to(device)
				test_pat_list.append(test_patterns)
			test_patterns = torch.cat(test_pat_list,dim=0)
			in_patterns = torch.cat([in_patterns, test_patterns], dim=0)
	
	# for high res, load high res patterns and network
	if opt.high_res:
		from scipy import signal

		opt.kernel_size = 9
		net = MyNet(opt, device).to(device)
		## initialization
		net.apply(weights_init)
		print(net)

		load_dict = torch.load(join(opt.load_ckpt,'ckpt.pt'))['net']

		for (src,dst) in zip(load_dict, net.named_parameters()):

			# print(dst[0], src)
			# dst[1].data = load_dict[src].data

			if load_dict[src].shape==dst[1].shape:
				dst[1].data = load_dict[src].data
			else:
				# upsampling (pytorch)
				sum_before = torch.sum(load_dict[src].data, dim=(-2,-1), keepdim=True)
				print('sum_before: ',sum_before.shape)
				src_up = nn.functional.interpolate(load_dict[src].data, size = 9, mode='bicubic')
				sum_after = torch.sum(src_up, dim=(-2,-1), keepdim=True)
				print('sum_after: ',sum_after.shape)


				dst[1].data = src_up*sum_before/sum_after
				# dst[1].data = load_dict[src].data
				# print(src_up.shape)

				# temp = signal.resample(load_dict[src].data.cpu().numpy(), [16,1,9,9])

				# print(temp.shape)

	else:
		net = MyNet(opt, device).to(device)
		net.apply(weights_init)
		print(net)

		# load model if needed
		if opt.load_ckpt != '':
			if 'net' in opt.load_ckpt_option:
				net.load_state_dict(torch.load(join(opt.load_ckpt,'ckpt.pt'))['net'])
				print(opt.load_ckpt, ' loaded network successfully!!')

	# edit if needed
	edit_img= None
	if opt.edit:
		edit_img = load_edit(opt)
		print('edit img: ', edit_img.shape)

	# perform optimization
	optim(opt, net, in_patterns, in_imgs, device, edit_imgs=edit_img)
