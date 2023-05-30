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

class Clustering_Dataset(Dataset):

	def __init__(self, df, data_path, opts, transform=None):
		self.df = df
		self.data_path = data_path
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		if self.opts.data == 'aptos' or 'aptos_test':
			image_id = self.df['id_code'][index]
			from_im = Image.open(f'{self.data_path}/{image_id}.png').convert('RGB')
		elif self.opts.data == 'eyepacs' or 'eyepacs_test':
			image_id = self.df['image'][index]
			from_im = Image.open(f'{self.data_path}/{image_id}.jpeg').convert('RGB')
		
		if self.transform:
			from_im = self.transform(from_im)
		
		return from_im, index, image_id

class InferenceDataset_for_centroid(Dataset):

	def __init__(self, root, opts, labels, transform=None):
		group_by_centers=labels.groupby(['centers'])

		self.lst = []
		if opts.data == 'aptos' or 'aptos_test':
			self.extension = '.png'
		elif opts.data == 'eyepacs' or 'eyepacs_test':
			self.extension = '.jpeg'

		for i in range(len(group_by_centers)):
			for j in np.asarray(group_by_centers['id_code'])[i][1]:
				path = os.path.join(root, j)
				fname = path+self.extension
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
