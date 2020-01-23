
import random
import numpy as np


### Setting hyperparameters
class Hyperparameters():
	def __init__(self):
		#
		self.batch_size					 = 64
		self.iterations					 = 1000
		self.lr							 = 1e-4
		self.alpha						 = 0.9
		self.sigma						 = 0.000000000000000001
		self.gain						 = 1.00
		self.no_init_filters			 = 64
		self.latent_size    			 = 100
		self.betas						 = 0.5

		#
		self.stats_frequency			 = 1000
		self.image_channel				 = 3
		self.image_size					 = 32
		
		#
		self.gpu	 					 = True
		self.verbose 					 = False

		#
		self.dataset_name				 = 'cifar32'
		self.source						 = 'mnist'
		self.target						 = 'usps'
		self.dataset_path 				 = '../../dataset/'
		self.model_mode					 = 'train'
		self.method						 = 'lcos'
		self.dataset_file_format		 = 'hdf5'
		self.dist_type					 = 'L1'
		self.experiment_name			 = 'test1'
		
		#
		self.no_total					 = 10
		self.no_closed					 =  6
		self.no_open 					 =  4

		#

		#
		self.tail_size 					 = 30
		self.open_alpha					 = self.no_closed
		self.dist_measure				 = 'euclidean'
		self.labels_name 				 = ['zero','one','two','three','four','five','six','seven','eight','nine']


		#
		self.HEADER		= '\033[95m'
		self.BLUE		= '\033[94m'
		self.GREEN		= '\033[92m'
		self.YELLOW		= '\033[93m'
		self.FAIL		= '\033[91m'
		self.ENDC		= '\033[0m'
		self.BOLD		= '\033[1m'
		self.UNDERLINE	= '\033[4m'





hyper_para = Hyperparameters()
