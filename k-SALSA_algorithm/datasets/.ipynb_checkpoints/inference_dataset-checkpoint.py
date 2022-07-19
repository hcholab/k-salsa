from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import os

class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		# print('from_path:',from_path)
		from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im, index, from_path

class new_EyePACS_InferenceDataset(Dataset):

	def __init__(self, df, data_path, opts, transform=None):
		# self.paths = sorted(data_utils.make_dataset(root))
		self.df = df
		self.data_path = data_path
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		image_id = self.df['image'][index]
		from_im = Image.open(f'{self.data_path}/{image_id}.jpeg').convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		
		return from_im, index, image_id

class InferenceDataset_for_centroid(Dataset):

	def __init__(self, root, opts, aptos_labels, transform=None):
		group_by_centers_aptos=aptos_labels.groupby(['centers'])

		self.lst = []
		for i in range(len(group_by_centers_aptos)):
			for j in np.asarray(group_by_centers_aptos['id_code'])[i][1]:
				path = os.path.join(root, j)
				fname = path+'.png'
				self.lst.append(fname) 
		# self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.lst)

	def __getitem__(self, index):
		from_path = self.lst[index]
		from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im, index, from_path #, from_path

class InferenceDataset_for_centroid_EyePACS(Dataset):

	def __init__(self, root, opts, eyepacs_labels, transform=None):
		group_by_centers_eyepacs=eyepacs_labels.groupby(['centers'])

		self.lst = []
		for i in range(len(group_by_centers_eyepacs)):
			for j in np.asarray(group_by_centers_eyepacs['id_code'])[i][1]:
				path = os.path.join(root, j)
				fname = path+'.jpeg'
				self.lst.append(fname) 

		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.lst)

	def __getitem__(self, index):
		from_path = self.lst[index]
		from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im, index, from_path