import sys
import os
import glob

import argparse


parser = argparse.ArgumentParser()
      

parser.add_argument('--res', type=int, default=512)        
parser.add_argument('--lr', type=float, default=0.02)        
parser.add_argument('--H_inten', type=float, default=1, help='height scale')        
parser.add_argument('--w_edit', type=float, default=10.0, help='weight of eidting term')        
parser.add_argument('--w_L1', type=float, default=0.1, help='weight of L1 term')        
parser.add_argument('--total_iter', type=int, default=2000)        
parser.add_argument('--seed', type=int, default=0)        
parser.add_argument('--name_pf', type=str, default='', help='')
parser.add_argument('--load_pf', type=str, default='', help='')
parser.add_argument('--loss', type=str, default='TD+16L1', help='loss function: TD+16L1')
parser.add_argument('--target', type=str, default='wood', help='save files directory')        
parser.add_argument('--ckpt_dir', type=str, default='wood', help='save files directory')        
parser.add_argument('--run_option', type=str, default='opt_ed_te', help='opt | ed | opt_ed || opt_ed_te')        
parser.add_argument('--in_pat_path', type=str, default='./data/Patterns2', help='the path for input patters') 
parser.add_argument('--in_img_path', type=str, default='./data/MyReal2', help='the path for input images') 
parser.add_argument('--load_ckpt_option', type=str, default='NA', help=' NA | net | pat | netpat || NA: no load ckpt; net: load network; pat: load pattern only; netpat: load both net and pat')        


parser.add_argument('--edit', action='store_true', help='save files directory')        
parser.add_argument('--scale_opt', action='store_true', help='optimize height scale, light intensity as well') 
parser.add_argument('--test', action='store_true', help='test or not')        
parser.add_argument('--no_optim_height', action='store_true', help='test or not')        
parser.add_argument('--use_tanh', action='store_true', help='use tanh')        
parser.add_argument('--decay_lr', action='store_true', help='use tanh')        
parser.add_argument('--only_edit', action='store_true', help='only editting')        
parser.add_argument('--resume', action='store_true', help='resume')        
parser.add_argument('--high_res', action='store_true', help='high_res')        
parser.add_argument('--tile', action='store_true', help='tile or not')        

# architecture
parser.add_argument('--pixconv_n', type=int, default=3)        
parser.add_argument('--ngf', type=int, default=16)        
parser.add_argument('--regconv', action='store_true', help='save files directory') 
parser.add_argument('--no_chconv', action='store_true', help='no_chconv') 


opt = parser.parse_args()

folder_path = opt.in_img_path

name_pf = 'regconv' if opt.regconv else opt.name_pf
load_pf = opt.name_pf if opt.load_pf=='' else opt.load_pf
scale_opt = ' --scale_opt' if opt.scale_opt else ''
regconv_str = ' --regconv ' if opt.regconv else ''
chconv_str = ' --add_channel_conv' if not opt.no_chconv and not opt.regconv else ''
optim = ' --no_optim_height' if opt.no_optim_height else ''
decay_lr = ' --decay_lr' if opt.decay_lr else ''
resume = ' --resume' if opt.resume else ''
high_res = ' --high_res' if opt.high_res else ''
tile = ' --tile' if opt.tile else ''

for idx, img in enumerate(os.listdir(folder_path)):

	print('img ', img)

	# repeat 5 times for each example
	for k in range(1):

		# optimization
		if 'opt' in opt.run_option:

			cmd = 'python train.py' \
				+ ' --checkpoints_dir ' + './ckpt/'+opt.ckpt_dir \
				+ ' --lr ' + str(opt.lr) \
				+ ' --res ' + str(opt.res) \
				+ ' --lambda_L1 ' + str(opt.w_L1) \
				+ ' --myclass NA ' \
				+ ' --name_pf ' + name_pf + '_rand_'+str(k) \
				+ ' --load_pf ' + load_pf + '_rand_'+str(k) \
				+ ' --load_option cust' \
				+ ' --in_pat_path ' + opt.in_pat_path \
				+ ' --load_ckpt_option ' + opt.load_ckpt_option \
				+ ' --pixconv_n ' + str(opt.pixconv_n) \
				+ ' --ngf ' + str(opt.ngf) \
				+ ' --output_nc ' + str(5) \
				+ ' --loss ' + opt.loss \
				+ ' --in_img ' + img \
				+ ' --real_root_path ' + folder_path \
				+ ' --total_iter ' + str(opt.total_iter) \
				+ ' --H_intensity ' + str(opt.H_inten) \
				+ ' --seed ' + str(opt.seed) \
				+ scale_opt + regconv_str + chconv_str + optim + decay_lr + resume + high_res + tile\

			print(cmd)
			os.system(cmd)

		# # test
		# if 'te' in opt.run_option:

		# 	cmd = 'python train.py' \
		# 		+ ' --checkpoints_dir ' + './ckpt/'+opt.ckpt_dir \
		# 		+ ' --folder ' + folder_name \
		# 		+ ' --lr ' + str(0.005) \
		# 		+ ' --res ' + str(opt.res) \
		# 		+ ' --lambda_L1 ' + str(opt.w_L1) \
		# 		+ ' --myclass ' + myclass \
		# 		+ ' --name_pf ' + name_pf + '_rand_'+str(k) \
		# 		+ ' --load_pf ' + load_pf + '_rand_'+str(k) \
		# 		+ ' --load_option ' + opt.load_option \
		# 		+ ' --load_ckpt_option netpat'\
		# 		+ ' --pixconv_n ' + str(opt.pixconv_n) \
		# 		+ ' --ngf ' + str(opt.ngf) \
		# 		+ ' --output_nc ' + str(5) \
		# 		+ ' --loss ' + opt.loss \
		# 		+ ' --in_img ' + img \
		# 		+ ' --real_root_path ' + folder_path \
		# 		+ ' --total_iter ' + str(opt.total_iter) \
		# 		+ ' --H_intensity ' + str(opt.H_inten) \
		# 		+ ' --total_iter ' + str(200) \
		# 		+ ' --save_freq ' + str(100) \
		# 		+ ' --seed ' + str(opt.seed) \
		# 		+ ' --test ' \
		# 		+ scale_opt + regconv_str + chconv_str + optim + decay_lr + resume + high_res\

		# 	print(cmd)
		# 	os.system(cmd)

		# # editing
		# if 'ed' in opt.run_option:

		# 	cmd = 'python train.py' \
		# 		+ ' --checkpoints_dir ' + './ckpt/'+opt.ckpt_dir \
		# 		+ ' --folder ' + folder_name \
		# 		+ ' --lr ' + str(opt.lr) \
		# 		+ ' --lambda_L1 ' + str(opt.w_L1) \
		# 		+ ' --w_edit ' + str(opt.w_edit) \
		# 		+ ' --res ' + str(opt.res) \
		# 		+ ' --real_root_path ' + folder_path \
		# 		+ ' --myclass ' + myclass \
		# 		+ ' --name_pf ' + name_pf + '_edit_rand_'+str(k) \
		# 		+ ' --load_pf ' + load_pf + '_rand_'+str(k) \
		# 		+ ' --load_option ' + opt.load_option \
		# 		+ ' --load_ckpt_option ' + opt.load_ckpt_option \
		# 		+ ' --pixconv_n ' + str(opt.pixconv_n) \
		# 		+ ' --ngf ' + str(opt.ngf) \
		# 		+ ' --output_nc ' + str(5) \
		# 		+ ' --loss ' + opt.loss \
		# 		+ ' --in_img ' + img \
		# 		+ ' --H_intensity ' + str(opt.H_inten) \
		# 		+ ' --total_iter ' + str(1000) \
		# 		+ ' --save_freq ' + str(50) \
		# 		+ ' --edit ' \
		# 		+ ' --seed ' + str(opt.seed) \
		# 	    + regconv_str + chconv_str + decay_lr + high_res\

		# 	print(cmd)
		# 	os.system(cmd)