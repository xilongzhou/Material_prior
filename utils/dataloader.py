import torch.utils.data as data

import os
from os import listdir
from os.path import join
import random

import imageio

import torch

from utils.util import *

class DataLoaderHelper(data.Dataset):
	def __init__(self, image_dir, opt):
		super(DataLoaderHelper, self).__init__()
		self.path = image_dir

		self.opt = opt
		self.image_filenames = [x for x in listdir(image_dir) if '.DS_Store' not in x]

	def _crop(self, img, pos, size):
		w = img.shape[0]
		h = img.shape[1]
		x1,y1=pos
		# print(w)
		if w > x1+size and h > y1+size:        
			return img[x1:x1+size, y1:y1+size,...]
		return img


	def __getitem__(self, index):
	   
		#load the image
		img_path = os.path.join(self.path,self.image_filenames[index]) 

		# resize $ cropping
		Crop_aug = bool(random.getrandbits(1))
		Rough_aug = bool(random.getrandbits(1))
		Roll_aug = bool(random.getrandbits(1))
		Rotat_aug = random.randint(0,4)

		crop_pos = np.random.randint(0,200,size=2)
		crop_size = random.randint(300,400)

		for img in listdir(img_path):

			key_name,ext = os.path.splitext(img)
			if ext not in ['.png', '.jpg', '.jpeg']:
				continue

			# print('debug ',img)
			temp_img = read_image(os.path.join(img_path, img))

			if Crop_aug and self.opt.aug_traindata:
				temp_img = self._crop(temp_img, crop_pos, crop_size)
				# print(temp_img.shape)

			if temp_img.shape[0] != self.opt.res:
				temp_img = torch.from_numpy(resize(temp_img, (self.opt.res, self.opt.res), anti_aliasing=True, mode='wrap'))
			else:		
				temp_img = torch.from_numpy(temp_img) #(W,H,C)

			## data augmentation
			if self.opt.aug_traindata:

				# rolling - X,Y
				roll_numX = random.randint(0,250)
				roll_numY = random.randint(0,250)
				temp_img = torch.roll(temp_img, shifts=(roll_numX, roll_numY), dims=(0, 1))

				# rotating & flip 
				# if Rotat_aug==0:
				# 	temp_img = torch.flip(temp_img,[1])
				# elif Rotat_aug==1:
				# 	temp_img = torch.flip(temp_img,[0])
				# elif Rotat_aug==2:
				# 	temp_img = torch.rot90(temp_img,3,[0,1])
				# elif Rotat_aug==3:
					# temp_img = torch.rot90(temp_img,1,[0,1])

				# color 
				rand_color = torch.rand(3)*0.8+0.1
				if '_diffuse.png' in img:
					# temp_img[:,:,0] = (temp_img[:,:,0]+rand_color[0])*0.5
					# temp_img[:,:,1] = (temp_img[:,:,1]+rand_color[1])*0.5
					# temp_img[:,:,2] = (temp_img[:,:,2]+rand_color[2])*0.5
					temp_img[:,:,0] = rand_color[0]
					temp_img[:,:,1] = rand_color[1]
					temp_img[:,:,2] = rand_color[2]
					
				# roughness
				rough_pertu = (torch.rand(1)*2-1)*0.05
				if '_roughness.png' in img and Rough_aug:
					temp_img = temp_img+rough_pertu				


			if '_diffuse.png' in img:
				diffuse = temp_img 
			elif '_normal.png' in img:
				normal = temp_img
			elif '_roughness.png' in img:
				rough = temp_img.unsqueeze(-1).repeat(1,1,3)
			elif '_metallic.png' in img:
				spec = temp_img.unsqueeze(-1).repeat(1,1,3)+0.04 ## add 0.04 to avoid 0

		return diffuse, normal, rough, spec

	def __len__(self):
		return len(self.image_filenames)




class DataLoaderHelper_inpat(data.Dataset):
	def __init__(self, image_dir, opt):
		super(DataLoaderHelper_inpat, self).__init__()
		self.path = image_dir
		self.image_filenames = [x for x in listdir(image_dir) if '.DS_Store' not in x]
		self.opt = opt
		self.images_number = len(listdir(os.path.join(self.path,self.image_filenames[0])))

	def __getitem__(self, index):

		#load the image
		folder_path = os.path.join(self.path,self.image_filenames[index]) 
		rand_index = random.randint(0,self.images_number-1)
		img_path = os.path.join(folder_path, self.image_filenames[index]+'_rand%d.png'%rand_index)
		# print(img_path)
		temp_img = read_image(img_path)

		if temp_img.shape[-1] != self.opt.res:
			temp_img = torch.from_numpy(resize(temp_img, (self.opt.res, self.opt.res), anti_aliasing=True))
		else:		
			temp_img = torch.from_numpy(temp_img)


		return temp_img

	def __len__(self):
		return len(self.image_filenames)