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
import os
import pickle

class Stage:
	'defining properties of a single experiment stage'
	def __init__(self, level, stage, corridors, next_stage, rule, condition, name, substages=0, random='pseudo'):
		self.level = level
		self.stage = stage
		self.corridors = corridors
		self.next_stage = next_stage
		self.rule = rule
		self.condition = condition
		self.name = name
		self.random = random
		self.N_corridors = len(corridors)
		self.substages = substages if substages else ['a'] * self.N_corridors


class StageCollection:
	'class for storing corridor properties'
	def __init__(self, image_path, experiment_name):
		self.image_path = image_path
		self.num_stages = 0
		self.name = experiment_name
		self.stages = []

	def add_stage(self, level, stage, corridors, next_stage, rule, condition, name, substages=0, random='pseudo'):
		self.stages.append(Stage(level, stage, corridors, next_stage, rule, condition, name, substages=substages, random=random))
		self.num_stages = self.num_stages + 1
