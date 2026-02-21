# -*- coding: utf-8 -*-
"""
Created in May 2019
@author: bbujfalussy - ubalazs317@gmail.com
A framework for storing virtual corridor properties for behavioral experiments
We define a class - Corridor - and the collector class.

"""

import numpy as np
from string import *
from sys import version_info
import io
import pickle

class Stage:
	"""defining properties of a single experiment stage"""
	def __init__(self, level, stage, corridors, next_stage, rule, condition, name, substages=0, random='pseudo'):
		self.level: str = level
		self.stage: int = stage
		self.corridors: list[int] = corridors
		self.next_stage: str = next_stage
		self.rule: str = rule
		self.condition: str = condition
		self.name: str = name
		self.random: str = random
		self.substages: list[str | int] = ['a'] * self.N_corridors if substages else substages

	@property
	def N_corridors(self):
		return len(self.corridors)


class StageCollection:
	"""class for storing corridor properties"""
	def __init__(self, image_path, experiment_name):
		self.image_path = image_path
		self.num_stages = 0
		self.name = experiment_name
		self.stages = []

	def add_stage(self, level, stage, corridors, next_stage, rule, condition, name, substages=0, random='pseudo'):
		self.stages.append(Stage(level, stage, corridors, next_stage, rule, condition, name, substages=substages, random=random))
		self.num_stages = self.num_stages + 1

	@staticmethod
	def from_json(file: io.TextIOWrapper) -> 'StageCollection':
		stage_collection = StageCollection(image_path=file['image_path'], experiment_name=file['name'])
		for stage in file['stages']:
			stage_collection.add_stage(
				level=stage['level'],
				stage=stage['stage'],
				corridors=stage['corridors'],
				next_stage=stage['next_stage'],
				rule=stage['rule'],
				condition=stage['condition'],
				name=stage['name'],
				substages=stage['substages'],
				random=stage['random'],
			)
		return stage_collection