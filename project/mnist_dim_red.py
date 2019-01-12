import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from sklearn.decomposition import PCA
import os
import numpy as np
import time

path = ''
#STANDARDIZATION; 100 epochs
#API: data = [X_train, Y_train, X_test, Y_test]
#============================================================================================================
def get_data():
	#Import dataset and normalize to [0,1]
	(data_train, labels_train), (data_test, labels_test) = mnist.load_data()
	data_train = data_train/255.0
	data_test = data_test/255.0

	data_train = data_train.astype('float32')
	data_test = data_test.astype('float32')

	#Flatten dataset (New shape for training and testing set is (60000,784) and (10000, 784))
	data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
	data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))

	#Create labels as one-hot vectors
	labels_train = keras.utils.np_utils.to_categorical(labels_train, num_classes=10)
	labels_test = keras.utils.np_utils.to_categorical(labels_test, num_classes=10)
	return [data_train, labels_train, data_test, labels_test]
#============================================================================================================
def gen_model(data, epochs, num_layers, fileName = None):
	'''generates and saves classifier trained on uncorrupted data'''
	[X_train, Y_train, X_test, Y_test] = data

	model = Sequential()
	model.add(Dense(100, activation='sigmoid', use_bias=True, kernel_initializer="normal", input_dim=num_layers))
	model.add(Dense(100, activation='sigmoid', kernel_initializer="normal"))
	model.add(Dense(10, activation='softmax', kernel_initializer="normal"))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=100, shuffle=True)
	model.save(fileName)
	return model
#============================================================================================================
def perturb(data, epochs = 10, epsilon = 0.25, order=np.inf): #nn architecture, attack
	'''trains its own model on train data, perturbs test data'''
	[X_train, Y_train, X_test, Y_test] = data
	#fileName = path + 'saved_models/adv_models/FGS/{}-100_{}epochs.h5'.format(784, epochs) #for semi semi white box case
	fileName = path + 'saved_models/pca_classifiers/{}.h5'.format(784) #classifier fileName
		
	#assert os.path.exists(fileName)
	if not os.path.exists(fileName):
		gen_model(data, epochs, 784, fileName)
	backend.set_learning_phase(False)
	model = keras.models.load_model(fileName)
	scores = model.evaluate(X_train, Y_train)
	sess =  backend.get_session()
	wrap = KerasModelWrapper(model)
	fgsm = FastGradientMethod(wrap, sess=sess)
	adv_x = fgsm.generate_np(X_test, eps=epsilon, ord = order, clip_min=0., clip_max=1.) #change ord to np.inf or 2, depending on fgs or fg
	return adv_x
#============================================================================================================
def perturb_arch_mismatch(data, epochs = 10, epsilon = 0.25, order=np.inf): #200-200-100-10
	'''trains its own model on train data, perturbs test data'''
	[X_train, Y_train, X_test, Y_test] = data
	fileName = path + 'saved_models/adv_models/arch_mismatch/200.h5'
	if not os.path.exists(fileName):
		model = Sequential()
		model.add(Dense(200, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=784))
		model.add(Dense(200, activation="relu", kernel_initializer="normal"))
		model.add(Dense(100, activation="relu", kernel_initializer="normal"))
		model.add(Dense(10, activation="softmax", kernel_initializer="normal"))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=200, shuffle=True)
		model.save(fileName)
			
	else:
		backend.set_learning_phase(False)
		model = keras.models.load_model(fileName)
	backend.set_learning_phase(False)
	sess =  backend.get_session()
	wrap = KerasModelWrapper(model)
	fgsm = FastGradientMethod(wrap, sess=sess)
	adv_x = fgsm.generate_np(X_test, eps=epsilon, ord=order,  clip_min=0., clip_max=1.)
	return adv_x
#============================================================================================================
def train_autoencoder(data, path_to_save, epochs = 100, num_layers = 80):
	'''path_to_save: path of .h5 file to store trained autoencoder in'''
	print('in train_autoencoder, num_layers = ', num_layers)
	#model.add(Dense(100, activation="relu", use_bias=True, kernel_initializer="normal", input_dim=num_layers))
	[X_train, Y_train, X_test, Y_test] = data
	autoencoder = Sequential()
	#autoencoder.add(Dense((784-num_layers)/2, input_dim=784, activation='relu'))
	#autoencoder.add(Dense(int(0.55*784), input_dim=784, activation='relu'))
	autoencoder.add(Dense(num_layers, input_dim = 784, activation='relu'))
	#autoencoder.add(Dense((784-num_layers)/2, activation='relu'))
	#autoencoder.add(Dense(int(0.55*784), activation='relu'))
	autoencoder.add(Dense(784, activation='sigmoid'))
	
	autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	autoencoder.fit(X_train, X_train, epochs=100, batch_size=500) #train
	autoencoder.save(path_to_save)
	return autoencoder
#============================================================================================================
def train_test_red_classifier(data, epochs = 10, num_layers = 392, epsilon = 0.25, order=np.inf, perturb_func=perturb):
	'''trains and tests classifier with input layer of reduced dimension using autoencoders'''
	[X_train, Y_train, X_test, Y_test] = data
	fileName = path + 'saved_models/classifiers/{}.h5'.format(num_layers) #classifier fileName
	if num_layers != 784:
		autoencoderFn = path + 'saved_models/autoencoders/{}_test.h5'.format(num_layers)
		if not os.path.exists(autoencoderFn):
			print('Training autoencoder ...')
			train_autoencoder(data, autoencoderFn, epochs = 100, num_layers = num_layers)
		backend.set_learning_phase(False)
		autoencoder = keras.models.load_model(autoencoderFn)
		get_hidden_layer_output = K.function([autoencoder.layers[0].input], [autoencoder.layers[0].output])
		red_dim = get_hidden_layer_output([X_train]) #input into classifier for training
		red_dim_test = get_hidden_layer_output([X_test]) #input into classifier for training
	else: #no defense
		red_dim = X_train; red_dim_test = X_test
	#-----------------------------------------------------------------------------------------------------------
	#FC 100-100-10
	if not os.path.exists(fileName):
		print('Training classifier ...')
		classifier = gen_model([red_dim, Y_train, red_dim_test, Y_test], epochs, num_layers, fileName)
	else:
		backend.set_learning_phase(False)
		classifier = keras.models.load_model(fileName)
		
	if epsilon == 0:
		perturbed_x = X_test
	else: #no perturbation
		perturbed_x = perturb_func(data, epochs, epsilon, order)
	if num_layers == 784:
		in_classifier = perturbed_x
	else: #no defense
		in_classifier = get_hidden_layer_output([perturbed_x]) #input into classifier for testing
	accuracy = classifier.evaluate(in_classifier, Y_test)
	return accuracy[1]*100
#============================================================================================================
def train_test_pca_classifier(data, epochs = 10, num_layers = 392, epsilon = 0.25, order =np.inf, perturb_func=perturb):
	'''trains and tests classifier with input layer of reduced dimension using PCA'''
	[X_train, Y_train, X_test, Y_test] = data
	fileName = path + 'saved_models/pca_classifiers/{}.h5'.format(num_layers) #classifier fileName
	if num_layers == 784:
		red_dim = X_train; red_dim_test = X_test
	else:
		pca=PCA(num_layers)
		red_dim = pca.fit_transform(X_train)
		red_dim_test = pca.transform(X_test)
	#-----------------------------------------------------------------------------------------------------------
	if not os.path.exists(fileName):
		print('Training classifier ...')
		classifier = gen_model([red_dim, Y_train, red_dim_test, Y_test], epochs, num_layers, fileName)
	else:
		backend.set_learning_phase(False)
		classifier = keras.models.load_model(fileName)
		
	if epsilon == 0:
		perturbed_x = X_test
	else:
		perturbed_x = perturb_func(data, epochs, epsilon, order)
	if num_layers == 784:
		in_classifier = perturbed_x
	else:
		in_classifier = pca.transform(perturbed_X)
	accuracy = classifier.evaluate(in_classifier, Y_test)
	return accuracy[1]*100
#============================================================================================================
def experiment(filePath, order = np.inf, perturb_func = perturb, func = train_test_red_classifier, epochs=100):
	num_layer_arr = [784, 331, 100, 80, 60, 40, 20]
	if order == np.inf:
		epsilon_arr = np.linspace(0.05,0.5, 10) #for FGS
	if order == 2:
		epsilon_arr = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5] #for FG

	data = get_data()
	for num_layers in num_layer_arr:
		since = time.time()
		accuracy = []; adv_succ = []
		to_write = ''
		baseline_accuracy = func(data, epochs = epochs, num_layers = num_layers, epsilon = 0, order=order, perturb_func = perturb_func) #no attack
		to_write += '{} NN attacked with epsilon = {}. Accuracy = {}, adversarial success = {}\n'.format(str(num_layers) + '-100-100-10', 0, baseline_accuracy, 0)
		for epsilon in epsilon_arr:
			acc = func(data, epochs = epochs, num_layers = num_layers, epsilon = epsilon, order=order, perturb_func=perturb_func)
			accuracy.append(acc)
			adv_succ.append((1.0 - (acc/baseline_accuracy))*100)
			to_write += '{} NN attacked with epsilon = {}. Accuracy = {}, adversarial success = {}\n'.format(str(num_layers) + '-100-100-10', epsilon, acc, (1.0 - (acc/baseline_accuracy))*100)
		print(to_write)
		with open(filePath, 'a') as fh:
			fh.write(to_write + '\n')
		print('Layer {} took {} seconds'.format(num_layers, time.time() - since))
#============================================================================================================
if __name__ == '__main__':
	#Autoencoder defense
	experiment('scenarios/wb_fgs/info.txt') #white box FGS
	experiment('scenarios/wb_fg/info.txt', order=2) #white box FG
	experiment('scenarios/bb_fgs/info.txt', order=np.inf, perturb_func = perturb_arch_mismatch) #black box FGS
	experiment('scenarios/bb_fg/info.txt', order=2, perturb_func = perturb_arch_mismatch) #black box FG
	'''#PCA defense
	experiment('scenarios/wb_fgs/info_pca.txt', order=np.inf, perturb_func=perturb, func=train_test_pca_classifier) #white box FGS
	experiment('scenarios/wb_fg/info_pca.txt', order=2, perturb_func=perturb, func=train_test_pca_classifier) #white box FG
	experiment('scenarios/bb_fgs/info_pca.txt', order=np.inf, perturb_func=perturb_arch_mismatch, func=train_test_pca_classifier) #black box FGS
	experiment('scenarios/bb_fg/info_pca.txt', order=2, perturb_func=perturb_arch_mismatch, func=train_test_pca_classifier) #black box FG'''
