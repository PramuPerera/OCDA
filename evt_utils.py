
import numpy as np
import sys



class Model:
	def __init__(self, flag, evt_init):
		self.flag = flag
		self.weibull_model = evt_init

def evt_train(data_raw, tail_size, max_min_flag):

	model = Model(max_min_flag, libmr.MR())

	if(model.flag=='reverse_weibull'):
		data_scores = data_raw
	elif(model.flag=='weibull'):
		data_scores = -data_raw
	else:
		print ("valid flags are max or min")

	# tail_size = len(data_raw)

	mr = libmr.MR()
	tail_data = sorted(data_scores)[:tail_size]
	mr.fit_high(tail_data, len(tail_data))
	model.weibull_model = mr

	# print tail_data
	# print model.weibull_model
	return model

def evt_train_v2(data_raw, tail_thr, max_min_flag):

	model = Model(max_min_flag, libmr.MR())

	if(model.flag=='reverse_weibull'):
		data_scores = data_raw
	elif(model.flag=='weibull'):
		data_scores = -data_raw
	else:
		print ("valid flags are max or min")

	mr = libmr.MR()
	sorted_data = sorted(data_scores)
	sorted_data = np.asarray(sorted_data)
	tail_data = sorted_data[sorted_data<=tail_thr]
	mr.fit_high(tail_data, np.shape(tail_data)[0])
	model.weibull_model = mr
	
	return model

def evt_test(evt_model, score_query):

	if(evt_model.flag=='reverse_weibull'):
		raw_score = score_query
	elif(evt_model.flag=='weibull'):
		raw_score = -score_query
	else:
		print ("valid flags are  max or min")
	
	evt_score = np.zeros(np.shape(score_query))
	for i in range(len(evt_score)):
		evt_score[i] = evt_model.weibull_model.w_score(raw_score[i])

	return evt_score