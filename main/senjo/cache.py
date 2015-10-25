import logging
logger = logging.getLogger('senjo')

import cv2
import numpy as np

import os
import json

from .helpers import DefaultJSONEncoder
class CacheEntry(object):
	def __init__(self,root_dir='', filename="descriptors.txt"):
		self.filename = filename
		self.root_dir = root_dir
		self._data = dict()
		self.is_dirty = False

	def __setitem__(self, k, v):
		self._data[k] = v
		self.is_dirty = True

	def __getitem__(self, k):
		try:
			return self._data[k]
		except KeyError:
			return None

class KeyPointsCacheEntry(CacheEntry):
	@staticmethod
	def deserialize(metadata):
		return KeyPointsCacheEntry()

	def serialize(self):
		pass

class DescriptorsCacheEntry(CacheEntry):
	def __init__(self, root_dir='', filename="descriptors.txt"):
		super(DescriptorsCacheEntry, self).__init__(root_dir=root_dir, filename=filename)
		self.root_dir = root_dir
		self.target_file = os.path.join(self.root_dir, self.filename)
		self._data = {}

	def to_json(self):
		self.serialize()
		return dict(
			filename=self.filename,
			root_dir=self.root_dir,
			data_file=self.target_file,
			data_mapping=self.data_mapping
		)

	@staticmethod
	def deserialize(metadata):
		filepath = metadata['data_file']
		data_mapping = metadata['data_mapping']
		filename = metadata['filename']
		root_dir = metadata['root_dir']

		data = np.loadtxt(filepath, dtype=np.float32)

		_data = {}
		for img_path, idx in data_mapping.items():
			_data[img_path] = data[idx]
		res = DescriptorsCacheEntry(filename, root_dir)
		res._data = _data
		return res

	def serialize(self, target_file=None):
		target_file = target_file or self.target_file
		logger.debug('writting %s' % target_file)

		data = []
		data_mapping = {}
		for i, item in enumerate(self._data.items()):
			k,v = item
			data_mapping[k] = i
			data.append(v)
		self.data_mapping = data_mapping
		data = np.vstack(data)
		np.savetxt(target_file, data)
		self.is_dirty = False

class Cache(object):
	def __init__(self, name=None, root_dir=None):
		if root_dir is not None:
			if not os.path.exists(root_dir):
				os.makedirs(root_dir)

		filename = '%s.json' % name
		filepath = os.path.join(root_dir, filename)

		logging.info('Using cache at %s' % filepath)

		self.name = name
		self.root_dir = root_dir
		self.filepath = filepath

		self.load()

	def __getitem__(self, k):
		return self._data[k]

	def to_json(self):
		return dict(
			name=self.name,
			descriptors=self._data['descriptors'],
			keypoints=self._data['keypoints']
		)

	def save(self, filename=None):
		filepath = filename or self.filepath

		if self._data['keypoints'].is_dirty or self._data['descriptors'].is_dirty:
			logger.info('saving cache to %s ..' % filepath)
			with open(filepath, 'wb') as fp:
				import json
				json.dump(self, fp, cls=DefaultJSONEncoder)
			self._data['keypoints'].serialize()
			self._data['descriptors'].serialize(target_file='%s-descriptors.txt' % self.name)
		else:
			logger.info('no changes to be saved.')

	def load(self, filename=None):
		filepath = filename or self.filepath
		if os.path.exists(filepath):
			logger.info('loading %s..' % filepath)
			with open(filepath, 'rb') as fp:
				self._data = json.load(fp)
			self._data['descriptors'] = DescriptorsCacheEntry.deserialize(self._data['descriptors'])
			self._data['keypoints'] = KeyPointsCacheEntry.deserialize(self._data['keypoints'])
		else:
			logger.info('new cache at %s' % filepath)

			self._data = dict(
				descriptors=DescriptorsCacheEntry(root_dir=self.root_dir, filename='%s-descriptors.txt' % self.name),
				keypoints=KeyPointsCacheEntry(root_dir=self.root_dir)
			)
