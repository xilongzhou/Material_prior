import torch 
import numpy as np
from utils.util import *


def GGX(cos_h, alpha):
	c2 = cos_h**2
	a2 = alpha**2
	den = c2 * a2 + (1 - c2)
	return a2 / (np.pi * den**2 + 1e-6)

def Fresnel(cos, f0):
	return f0 + (1 - f0) * (1 - cos)**5

def Fresnel_S(cos, specular):
	sphg = torch.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos);
	return specular + (1.0 - specular) * sphg

def Smith(n_dot_v, n_dot_l, alpha):
	def _G1(cos, k):
		return cos / (cos * (1.0 - k) + k)
	k = (alpha * 0.5).clamp(min=1e-6)
	return _G1(n_dot_v, k) * _G1(n_dot_l, k)

# def norm(vec): #[B,C,W,H]
# 	vec = vec.div(vec.norm(2.0, 1, keepdim=True))
# 	return vec

def getDir(pos, tex_pos):
	vec = pos - tex_pos
	return norm(vec), (vec**2).sum(1, keepdim=True)

# def AdotB(a, b):
# 	return (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)
def getTexPos(res, size, device):
	x = torch.arange(res, dtype=torch.float32)
	x = ((x + 0.5) / res - 0.5) * size

	# surface positions,
	y, x = torch.meshgrid((x, x))
	z = torch.zeros_like(x)
	pos = torch.stack((x, -y, z), 0).to(device)

	return pos

def render(maps, tex_pos, li_color, camli_pos, gamma=True, isSpecular=False):

	assert len(li_color.shape)==4, "dim of the shape of li_color pos should be 4"
	assert len(camli_pos.shape)==4, "dim of the shape of camlight pos should be 4"
	assert len(tex_pos.shape)==4, "dim of the shape of position map should be 4"
	assert len(maps.shape)==4, "dim of the shape of feature map should be 4"
	assert camli_pos.shape[1]==3, "the 1 channel of position map should be 3"

	normal = maps[:,0:3,:,:]
	albedo = maps[:,3:6,:,:]
	rough = maps[:,6:9,:,:]
	if isSpecular:
		specular = maps[:,9:12,:,:]
	else:
		f0 = maps[:,9:12,:,:]*0.0+0.04

	v, _ = getDir(camli_pos, tex_pos)
	l, dist_l_sq = getDir(camli_pos, tex_pos)
	h = norm(l + v)
	normal = norm(normal)


	n_dot_v = AdotB(normal, v)
	n_dot_l = AdotB(normal, l)
	n_dot_h = AdotB(normal, h)
	v_dot_h = AdotB(v, h)

	# print('dist_l_sq:',dist_l_sq)
	geom = n_dot_l / (dist_l_sq + eps)
	# geom = n_dot_l / 16

	D = GGX(n_dot_h, rough**2)

	if isSpecular:
		F = Fresnel_S(v_dot_h, specular)
	else:
		F = Fresnel(v_dot_h, f0)

	G = Smith(n_dot_v, n_dot_l, rough**2)

	## lambert brdf
	f1 = albedo / np.pi

	if isSpecular:
		f1 = f1*(1-specular)

	## cook-torrance brdf
	f2 = D * F * G / (4 * n_dot_v * n_dot_l + eps)

	f = f1 + f2

	img = f * geom * li_color

	if gamma:
		return img.clamp(eps, 1.0)**(1/2.2)		
	else:
		return img.clamp(eps, 1.0)


def affine_img(opt, x, scale_factor, device):

	light, light_pos, size = set_params(opt, device)
	tex_pos = getTexPos(opt.res, size, device).unsqueeze(0)

	_,_,w,h = x.shape

	x = x.tile((1,1,3,3)).to(device)

	# [opt.ngf, 2] --> [opt.ngf, 2, 3]
	theta_temp = torch.zeros(1,6).to(device)
	theta_temp[:,0] = scale_factor
	theta_temp[:,4] = scale_factor
	in_theta = theta_temp.view(-1,2,3)

	grid = F.affine_grid(in_theta,x.size())
	x = F.grid_sample(x, grid)
		
	x = x[:,:,w:2*w,h:2*h]

	normal = x[:,0:3,:,:]*2-1
	albedo = x[:,3:6,:,:]**2.2
	rough = x[:,6:7,:,:].expand(-1,3,-1,-1)
	specular = torch.tensor([0.04]).expand(x.shape[0], 3, x.shape[2], x.shape[3]).to(device)

	out_maps = torch.cat((normal, albedo, rough, specular), dim=1)

	out_ren = render(out_maps, tex_pos, light, light_pos, device)

	return out_ren

if __name__ == '__main__':
	pass

