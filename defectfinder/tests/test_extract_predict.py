import numpy as np
from scipy import fftpack
from scipy import ndimage
from sklearn.preprocessing import OneHotEncoder
import cv2
from defectfinder import extract_predict1 as ext

""" Testing functions for extract_predict modules"""

def  test_FFTmask():
	try:
		imgsrc= ()
		maskratio= int
		ext.FFTmask(imgsrc, maskratio)

	except(Exception):
		pass
	else:
		raise Exception('Error not handled')
	return


def test_FFTsub():
	try:
		imgsrc=()
		F3=()
		ext.FFTsub(imgsrc, F3)
	except(Exception):
		pass
	else:
		raise Exception('Error not handled')
	return


def test_threshImg():
	try:
		diff=np.float
		threshL=np.float
		threshH=np.float
		ext.threshImg(diff, threshL, threshH)
	except(Exception):
		pass
	
	else:
		raise Exception('Error not handled')
	return


def test_GenerateXYPos():
	try:
		window_size=int
		window_step=int
		image_width=int
		ext.GenerateXYPos(window_size, window_step, image_width)
	except(Exception):
		pass
	
	else:
		raise Exception('Error not handled')
	return

def test_MakeWindow():
	try:
		imgsrc=()
		xpos=int
		ypos=int
		window_size=int
		ext.MakeWindow(imgsrc, xpos, ypos, window_size)
	except(Exception):
		pass
	
	else:
		raise Exception('Error not handled')
	return

def test_imgen():
	try:
		raw_image=()
		pos_mat=()
		window_size=int
		ext.imgen(raw_image,pos_mat,window_size)
	except(Exception):
		pass
	
	else:
		raise Exception('Error not handled')
	return



def test_image_preprocessing():
	try:
		image_data=()
		norm=int
		ext.image_preprocessing(image_data, norm)
	except(Exception):
		pass
	
	else:
		raise Exception('Error not handled')
	return


def test_label_preprocessing():
	try:
		image_data=()
		nb_classes=int
		ext.label_preprocessing(image_data, nb_classes)
	except(Exception):
		pass
	
	else:
		raise Exception('Error not handled')
	return


def test_label_preprocessing2():
	try:
		image_data=()
		ext.label_preprocessing2(image_data)
	except(Exception):
		pass
	
	else:
		raise Exception('Error not handled')
	return


def test_splitImage():
	try:
		img=()
		target_size=int
		ext.splitImage(img, target_size)
	except(Exception):
		pass
	
	else:
		raise Exception('Error not handled')
	return

def test_predictDefects():
	try:
		img=()
		model= keras.models
		target_size=int
		nb_classes=int
		ext.predictDefects(img, model, target_size, nb_classes)
	except(Exception):
		pass
	
	else:
		raise Exception('Error not handled')
	return


def test_extractDefects():
	try:
		img=()
		classpred=str 
		softmax_threhold=np.float
		bbox=int
		ext.extractDefects(img, classpred, softmax_threhold, bbox)
	except(Exception):
		pass
	
	else:
		raise Exception('Error not handled')
	return



