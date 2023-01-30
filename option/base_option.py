import argparse
import os
import random
from utils.util import *
from os.path import join

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):  

		################## input ##################
		self.parser.add_argument('--res', type=int, default=512, help='the resolution of an image')        
		self.parser.add_argument('--rand_pat', type=int, default=-1, help='randomize input patterns or not || 0: no random; 1,2,...: random')        
		self.parser.add_argument('--in_pat_type', type=int, default=-1, help='combination # of different input patterns')        
		self.parser.add_argument('--in_img', type=str, default='in.png', help='input image name')        
		self.parser.add_argument('--myclass', type=str, default='', help='loaded based on class')        
		self.parser.add_argument('--load_option', type=str, default='', help='class | rand || load pattern based on class or rand')        
		self.parser.add_argument('--load_ckpt_option', type=str, default='', help=' net || pat || netpat')        
		self.parser.add_argument('--real_root_path', type=str, default='', help='real_root_path')        
		self.parser.add_argument('--in_pat_path', type=str, default='', help='path of pat')        
		self.parser.add_argument('--aug_inpats', action='store_true', help='augmentation of inputs or not')
		self.parser.add_argument('--aug_traindata', action='store_true', help='augmentation of trainning dataset of encoder or not')
		self.parser.add_argument('--batch_size', type=int, default=1, help='the batch size of training the encoder')        
		self.parser.add_argument('--N_class', type=int, default=2, help='Number of class patterns')        
		self.parser.add_argument('--N_common', type=int, default=3, help='Number of common patterns')        


		################## output ##################
		self.parser.add_argument('--save_gif_freq', type=int, default=10, help='log and saved per # iters') 
		self.parser.add_argument('--save_ckpt_freq', type=int, default=1000, help='# epoch to save ckpt') 
		self.parser.add_argument('--FPS', type=int, default=4, help='FPS of saving gif') 
		self.parser.add_argument('--save_gif', action='store_true', help='save gif to not')        
		self.parser.add_argument('--vis_interlayer', action='store_true', help='visualize intermeidate layer or not')
		self.parser.add_argument('--load_ckpt', type=str, default='', help='path of loaded ckpt')        
		self.parser.add_argument('--load_iter', type=str, default='last', help='iteration of loaded ckpt')        


		################## training settings ##################
		self.parser.add_argument('--seed', type=int, default=0, help='random seed:  902434748 (fail), 2109923418(good)')        
		self.parser.add_argument('--gpu', type=int, default=0, help='0,1,2,3.... | -1: fpr cpu')
		self.parser.add_argument('--logs', action='store_true', help='logs or not')        
		self.parser.add_argument('--STN_lr', type=float, default=0.02, help='learning rate of STN') 
		self.parser.add_argument('--decay_lr', action='store_true', help='decaying learning rate')        
		self.parser.add_argument('--lr', type=float, default=0.01, help='learning rate || 0.01 : network w/o estimation part || 0.0001: with estimation network') 
		self.parser.add_argument('--lambda_L1', type=float, default=0.1, help='lambda of L1 loss') 
		self.parser.add_argument('--lambda_TD', type=float, default=1, help='lambda of TD loss') 
		self.parser.add_argument('--lambda_KLD', type=float, default=0.1, help='lambda of KLD loss') 
		self.parser.add_argument('--lambda_smooth', type=float, default=1, help='lambda of smooth loss') 
		self.parser.add_argument('--lambda_theta_STN', type=float, default=0, help='lambda of theta STN loss') 
		self.parser.add_argument('--H_intensity', type=float, default=3.0, help='intensity') 
		self.parser.add_argument('--w_edit', type=float, default=10.0, help='weight of editing') 
		self.parser.add_argument('--TD_pyramid', type=int, default=0, help='use pyramid or not for TD loss') 
		self.parser.add_argument('--save_freq', type=int, default=1000, help='log and saved per # iters') 
		self.parser.add_argument('--total_iter', type=int, default=2000, help='total number of iterations') 
		self.parser.add_argument('--loss', type=str, default='TD+L1', help='TD: texture descriptor (VGG19) || L1 || L2')        
		self.parser.add_argument('--Optim_pat', action='store_true', help='optim input pattern or not')        
		self.parser.add_argument('--multi_scale', action='store_true', help='multi scaling optimization or not')        
		self.parser.add_argument('--add_smooth_loss', action='store_true', help='add smooth loss or not')        
		self.parser.add_argument('--freeze_MLP', action='store_true', help='freeze the MLP of learned network')        
		self.parser.add_argument('--test', action='store_true', help='freeze the MLP of learned network')        
		self.parser.add_argument('--scale_opt', action='store_true', help='optimize light intensity or not')        
		self.parser.add_argument('--no_optim_height', action='store_true', help='not optimize height intensity or not')        
		self.parser.add_argument('--load_net_only', action='store_true', help='only load network')        
		self.parser.add_argument('--resume', action='store_true', help='resume or not')        
		self.parser.add_argument('--high_res', action='store_true', help='high_res or not')        

		#### train encoder
		self.parser.add_argument('--Train_Encoder', action='store_true', help='train the encoder')        
		# self.parser.add_argument('--train_enco_pattern', type=str, default='./Dataset/materialsRendered/Brick_inpat_select', help='the path of input pattens')        
		self.parser.add_argument('--train_enco_pattern', type=str, default='./Dataset/materialsRendered/Leather_inpat_select', help='the path of input pattens')        
		self.parser.add_argument('--train_enco_img', type=str, default='./Dataset/materialsRendered/Leather_feren', help='the path of input images')        

		################## testing ##################
		self.parser.add_argument('--load_latent', action='store_true', help='load the optimized latent')        
		self.parser.add_argument('--edit', action='store_true', help='testing or not with different input patterns')        


		################## architecture ##################
		self.parser.add_argument('--pixconv_n', type=int, default=3, help='the number of 1x1xmxn conv layers (per pixel operation)') 
		self.parser.add_argument('--STN_use', type=str, default='NA', help='use STN or not || NA: no STN || expl: explcit TN || net: network STN') 
		self.parser.add_argument('--STN_theta', type=str, default='s1', help='s1: iso scale || s2: aniso scale') 
		self.parser.add_argument('--STN_type', type=str, default='all', help='all: all feature maps share (2,3) parameters || sep: each feature maps has own (2,3) parameters')        
		self.parser.add_argument('--order', type=str, default='STN_pcon', help='STN_pcon || pcon_STN')        
		self.parser.add_argument('--STN_pad', type=str, default='circular', help='padding for Spatial Transformer Network: reflection || zeros')        
		self.parser.add_argument('--only_STN', action='store_true', help='only use STN layer') 
		self.parser.add_argument('--add_STN_last', action='store_true', help='add STN to the last layer') 
		self.parser.add_argument('--add_channel_conv', action='store_true', help='add channel convolution or not') 
		self.parser.add_argument('--input_nc', type=int, default=5, help='the number of input channel in the first layer)') 
		self.parser.add_argument('--ngf', type=int, default=16, help='# of channels in the first layer') 
		self.parser.add_argument('--kernel_size', type=int, default=5, help='size of kernel') 
		self.parser.add_argument('--output_nc', type=int, default=5, help='9: normal, albedo, rough, spec || 7: normal, albdeo, rough, metallic || 8: height, albdeo, rough, spec || 5: height, albedo, rough || 4: albedo, rough') 
		self.parser.add_argument('--bias', action='store_true', help='with bias or not')        
		self.parser.add_argument('--regconv', action='store_true', help='regular conv or not')        
		self.parser.add_argument('--use_tanh', action='store_true', help='use tanh for the last output layer')        
		self.parser.add_argument('--n_mlp', type=int, default=8, help='# of MLP') 
		self.parser.add_argument('--Sample_Net', type=str, default='', help='samplenet type: 1: heated || 2: pixconv')        


		#### training network to estimate params
		self.parser.add_argument('--net_params', action='store_true', help='train network to estimate the parameters of network')        
		self.parser.add_argument('--pnet_enco', action='store_true', help='encoder network')        
		self.parser.add_argument('--ngf2', type=int, default=16, help='# of channels in the first conv layer of estimate network') 
		self.parser.add_argument('--MLP_nc', type=int, default=512, help='# of channels in the first MLP layer of estimation network') 

		### discriminator 
		self.parser.add_argument('--ndf', type=int, default=16, help='# of channels in discriminator') 
		self.parser.add_argument('--ndl', type=int, default=3, help='# of layers in discriminator') 
		self.parser.add_argument('--lambda_GP', type=float, default=1.0, help='the lambda of GP term') 


		################## path ##################       
		self.parser.add_argument('--checkpoints_dir', type=str, default='ckpt', help='the directory of saving checkpoints')        
		self.parser.add_argument('--log_dir', type=str, default='./logs', help='log directory')        
		self.parser.add_argument('--folder', type=str, default='', help='the name of scene')        
		self.parser.add_argument('--name_pf', type=str, default='', help='the name postfix of subfolder in certain texture')        
		self.parser.add_argument('--load_pf', type=str, default='', help='the name postfix of subfolder in certain texture')        


		################## debug settings ##################
		self.parser.add_argument('--debug_render', action='store_true', help='debug render or not')
		self.parser.add_argument('--debug_STN', action='store_true', help='debug STN or not')
		self.parser.add_argument('--save_per_batch', action='store_true', help='save per batch or not')
		self.parser.add_argument('--tile', action='store_true', help='save per batch or not')


		self.initialized = True

	def parse(self,save=True):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		args = vars(self.opt)

		loss_list = self.opt.loss.split('+')

		if self.opt.lambda_TD<1:
			temp_lambda_TD = '1e-1'
		else:
			temp_lambda_TD = self.opt.lambda_TD

		if self.opt.lambda_L1<1:
			temp_lambda_L1 = '1e-1'
		else:
			temp_lambda_L1 = self.opt.lambda_L1

		loss_name=''
		for name_index, temp_name in enumerate(loss_list):

			if 'TD' in temp_name:
				new_name = (str(temp_lambda_TD) if self.opt.lambda_TD !=1 else '') + temp_name
			elif 'L1' in temp_name:
				new_name = (str(temp_lambda_L1) if self.opt.lambda_L1 !=1 else '') + temp_name
			elif 'GAN' in temp_name:
				new_name = 'GAN'

			print(new_name)
			loss_name += new_name + ('' if name_index==len(loss_list)-1 else '+')

		if not self.opt.Train_Encoder:
			STN_type = 'a' if self.opt.STN_type=='all' else 's'

			# self.opt.name2 = "%s_%dl%dc%s%s_%d%sin%do" % (loss_name, self.opt.pixconv_n, self.opt.ngf,(str(self.opt.kernel_size)+'k') if self.opt.add_channel_conv else '', '%sSTN%s'%(STN_type,self.opt.STN_theta) if self.opt.STN_use!='NA' else '' ,self.opt.input_nc, 'rand'+str(self.opt.rand_pat) if self.opt.rand_pat>0 else '', self.opt.output_nc)
			self.opt.name2 = "%s_%dl%dc%s%s_%d%sin%do" % (loss_name, self.opt.pixconv_n, self.opt.ngf,(str(self.opt.kernel_size)+'k'), '%sSTN%s'%(STN_type,self.opt.STN_theta) if self.opt.STN_use!='NA' else '' ,self.opt.input_nc, 'rand'+str(self.opt.rand_pat) if self.opt.rand_pat>0 else '', self.opt.output_nc)
			self.opt.name2 = join(self.opt.folder,self.opt.name2)
		else:
			self.opt.name2 = join(self.opt.folder,"pretrained_%s"%self.opt.name_pf)

		if self.opt.seed ==0:
			self.opt.seed = random.randint(0, 2**31 - 1)

		# print('------------ Options -------------')
		# for k, v in sorted(args.items()):
		# 	print('%s: %s' % (str(k), str(v)))
		# # print('seed %s' % str(seed))
		# print('-------------- End ----------------')


		if save:# and not self.opt.continue_train:
			# save to the disk        
			expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.myclass, self.opt.name2+ '_'+self.opt.name_pf)
			mkdirs(expr_dir)

			file_name = os.path.join(expr_dir, 'opt.txt')
			with open(file_name, 'wt') as opt_file:
				opt_file.write('------------ Options -------------\n')
				for k, v in sorted(args.items()):
					opt_file.write('%s: %s\n' % (str(k), str(v)))
				# opt_file.write('seed %s\n' % str(seed))
				opt_file.write('-------------- End ----------------\n')

		return self.opt

