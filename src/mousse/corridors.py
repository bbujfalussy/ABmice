import io

import numpy as np
from string import *
from sys import version_info
import os
import pickle

class Corridor:
	"""defining properties of a single corridor"""
	def __init__(
			self,
			name: str,
			left_image: str,
			right_image: str,
			end_image: str,
			floor_image: str,
			ceiling_image: str,
			reward_zone_starts: list[int],
			zone_width=470,
			reward='Right',
			length=7168,
			height=768,
			width=1024
	):
		self.name = name

		self.left_image = left_image
		self.right_image = right_image
		self.end_image = end_image
		self.floor_image = floor_image
		self.ceiling_image = ceiling_image

		self.length = length
		self.height = height
		self.width = width


		# zone_shift = 0.0233 # Rita wants the zones to start at the near-edge of the monitor
		self.reward = reward # currently all zones in a given corridor are identical. In the future, we could have a vector for encoding different zone properties
		self.N_zones = len(reward_zone_starts)

		zone_shift = 0 ## 0.05 for Rita
		section_length = float(self.length - self.width)
		self.reward_zone_starts = np.array(reward_zone_starts) / section_length + zone_shift # relative position of reward zone starts [0, 1]
		self.reward_zone_ends = self.reward_zone_starts + zone_width / section_length

		if self.N_zones > 0:
			for i in np.arange(self.N_zones):
				if self.reward_zone_starts[i] < 0:
					self.reward_zone_starts[i] = 0
				if self.reward_zone_starts[i] > 1:
					self.reward_zone_starts[i] = 1
				if self.reward_zone_ends[i] < self.reward_zone_starts[i]:
					self.reward_zone_ends[i] = self.reward_zone_starts[i]
				if self.reward_zone_ends[i] > 1:
					self.reward_zone_ends[i] = 1


class CorridorCollection:
	"""class for storing corridor properties"""
	def __init__(self, image_path, experiment_name):
		self.image_path = image_path
		self.name = experiment_name
		self.corridors = []

	@property
	def num_VRs(self):
		return len(self.corridors)

	def add_corridor(self, name, left_image, right_image, end_image, floor_image, ceiling_image, reward_zone_starts, zone_width=470, reward='Right', length=7168, height=768, width=1024):
		self.corridors.append(Corridor(name, left_image, right_image, end_image, floor_image, ceiling_image, reward_zone_starts, zone_width, reward, length, height, width))

	@staticmethod
	def from_json(file: io.TextIOWrapper) -> 'CorridorCollection':
		stage_collection = CorridorCollection(image_path=file['image_path'], experiment_name=file['name'])
		for stage in file['corridors']:
			stage_collection.add_corridor(
				name=stage['name'],
				left_image=stage['left_image'],
				right_image=stage['right_image'],
				end_image=stage['end_image'],
				floor_image=stage['floor_image'],
				ceiling_image=stage['ceiling_image'],
				reward_zone_starts=stage['reward_zone_starts'],
				width=stage['width'],
				length=stage['length'],
				height=stage['height'],
				reward=stage['reward'],
			)
		return stage_collection
