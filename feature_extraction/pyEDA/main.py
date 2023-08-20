# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import os

# Importing necessary libraries
import numpy as np
import scipy.signal as sps


def resample_data(gsrdata, prevSR, newSR):
  '''calculates rolling mean
    Function to calculate moving average over the passed data
	
    Parameters
    ----------
    gsrdata : 1-d array
        array containing the gsr data
    prevSR : int or float 
        the previous sample rate of the data
    newSR : int or float
        the new sample rate of the data
		
    Returns
    -------
    data : 1-d array
        array containing the resampled data
  '''
  number_of_samples = int(round(len(gsrdata) * float(newSR) / prevSR))
  data = sps.resample(gsrdata, number_of_samples)
  
  return data

	
def normalization(gsrdata):
  '''min max normalization
    Function to calculate normalized gsr data
	
    Parameters
    ----------
    gsrdata : 1-d array
        array containing the gsr data
		
    Returns
    -------
    n_gsrdata : 1-d array
        normalized gsr data
  '''
  gsrdata = gsrdata-(np.min(gsrdata))
  gsrdata /= (np.max(gsrdata) - np.min(gsrdata))
  n_gsrdata = gsrdata
  return n_gsrdata

def rolling_mean(data, windowsize, sample_rate):
  '''calculates rolling mean
    Function to calculate moving average over the passed data
	
    Parameters
    ----------
    data : 1-d array
        array containing the gsr data
    windowsize : int or float 
        the moving average window size in seconds 
    sample_rate : int or float
        the sample rate of the data set
		
    Returns
    -------
    rol_mean : 1-d array
        array containing computed rolling mean
  '''
  avg_hr = (np.mean(data))
  data_arr = np.array(data)
	
  t_windowsize = int(windowsize*sample_rate)
  t_shape = data_arr.shape[:-1] + (data_arr.shape[-1] - t_windowsize + 1, t_windowsize)
  t_strides = data_arr.strides + (data_arr.strides[-1],)
  sep_win = np.lib.stride_tricks.as_strided(data_arr, shape=t_shape, strides=t_strides)
  rol_mean = np.mean(sep_win, axis=1)
	
  missing_vals = np.array([avg_hr for i in range(0, int(abs(len(data_arr) - len(rol_mean))/2))])
  rol_mean = np.insert(rol_mean, 0, missing_vals)
  rol_mean = np.append(rol_mean, missing_vals)

  #only to catch length errors that sometimes unexplicably occur. 
  ##Generally not executed, excluded from testing and coverage
  if len(rol_mean) != len(data): # pragma: no cover
    lendiff = len(rol_mean) - len(data)
    if lendiff < 0:
      rol_mean = np.append(rol_mean, 0)
    else:
      rol_mean = rol_mean[:-1]
	  
  return rol_mean


import torch
import numpy as np
from torch import nn
import math


class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        closest_pow2 = pow(2,int(math.floor(math.log(kwargs["input_shape"],2))))
		
		# Encoder layers
        self.linear1 = nn.Linear(in_features=kwargs["input_shape"], out_features=closest_pow2)
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 16, 3, padding=1)
        self.maxpool = nn.MaxPool1d(2, 2)
        self.linear2 = nn.Linear(in_features=(closest_pow2)*2, out_features=kwargs["latent_size"])
		
		# Decoder layers
        self.linear3 = nn.Linear(in_features=kwargs["latent_size"], out_features=(closest_pow2)*2)
        self.deconv1 = nn.ConvTranspose1d(16, 32, 2, stride=2)
        self.deconv2 = nn.ConvTranspose1d(32, 64, 2, stride=2)
        self.deconv3 = nn.ConvTranspose1d(64, 1, 2, stride=2)
        self.linear4 = nn.Linear(in_features=closest_pow2, out_features=kwargs["input_shape"])
		
        
    def forward(self, features):
		# Encoder
        activation = self.linear1(features)
        activation = torch.reshape(activation, (activation.shape[0],1,activation.shape[1]))
        activation = self.conv1(activation)
        activation = torch.relu(activation)
        activation = self.maxpool(activation)
        activation = self.conv2(activation)
        activation = torch.relu(activation)
        activation = self.maxpool(activation)
        activation = self.conv3(activation)
        activation = torch.relu(activation)
        activation = self.maxpool(activation)
        d = activation.shape
        activation = torch.reshape(activation, (d[0],d[1]*d[2]))
        code = self.linear2(activation)

		# Decoder
        activation = self.linear3(code)
        activation = torch.reshape(activation, (d[0],d[1],d[2]))
        activation = self.deconv1(activation)
        activation = torch.relu(activation)
        activation = self.deconv2(activation)
        activation = torch.relu(activation)
        activation = self.deconv3(activation)
        activation = torch.sigmoid(activation)
        activation = torch.reshape(activation, (activation.shape[0],activation.shape[2]))
        reconstructed = self.linear4(activation)
		
        return reconstructed, code
		

def create_train_loader(gsrData, batch_size=10):
	train_loader = []
	tensor_data = []
	
	for data in gsrData:
		tensor_data.append(np.array(data).flatten())
		if (len(tensor_data) == batch_size):
			train_loader.append(tensor_data)
			tensor_data = []

	if (len(tensor_data) != 0):
		print("Train data concatenated due to incompatible batch_size!")
	
	return torch.FloatTensor(train_loader)



    
def prepare_automatic(gsr_signal, sample_rate=128, new_sample_rate=20, k=32, epochs=10, batch_size=1, model_path=None):
    gsrdata = np.array(gsr_signal).T
    print("If you are using this tool for your research please cite this paper: \"pyEDA: An Open-Source Python Toolkit for Pre-processing and Feature Extraction of Electrodermal Activity\"");
    
    #################################################################################
    ############################## Preprocessing Part ###############################
    
    # Resample the data based on original data rate of your device, here: 128Hz + rolling window
    
    preprocessed_gsr = []
    #for i in gsrdata:
    data = resample_data(gsrdata, sample_rate, new_sample_rate)
    preprocessed_gsr.append(rolling_mean(data, 1./new_sample_rate, new_sample_rate))
    preprocessed_gsr = np.array(gsrdata).reshape(1,gsrdata.shape[0])
    print(preprocessed_gsr.shape)
    
    ############################## Preprocessing Part ###############################
    #################################################################################
    
    
    #################################################################################
    ############################ Train the Autoencoder ##############################
    
    # set the input shape to model
    input_shape = preprocessed_gsr.shape[1]
    
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=input_shape, latent_size=k).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()
    
    # create tensor data
    train_loader = create_train_loader(preprocessed_gsr, batch_size)
    print(train_loader.shape)
    
    # Training the network
    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            # compute reconstructions
            outputs,_ = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            
        # compute the epoch training loss
        loss = loss / len(train_loader)
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        
        
    torch.save(model, model_path)
        
	############################ Train the Autoencoder ##############################
	#################################################################################
    train_outputs, latent_variable = model(torch.FloatTensor(preprocessed_gsr))
    return latent_variable.detach().numpy()[0];

	

def process_automatic(gsr_signal, model_path):
	#################################################################################
	############################ Feature Extraction Part ############################
    gsrdata = np.array(gsr_signal)
    print("If you are using this tool for your research please cite this paper: \"pyEDA: An Open-Source Python Toolkit for Pre-processing and Feature Extraction of Electrodermal Activity\"");
    
    #################################################################################
    ############################## Preprocessing Part ###############################
    
    # Resample the data based on original data rate of your device, here: 128Hz + rolling window
    
    preprocessed_gsr = []
    sample_rate=512
    new_sample_rate = 512
    data = resample_data(gsr_signal, sample_rate, new_sample_rate)
    preprocessed_gsr.append(rolling_mean(data, 1./new_sample_rate, new_sample_rate))
    preprocessed_gsr = np.array(gsrdata).reshape(1,gsrdata.shape[0])
    
    
    

    model = torch.load(os.path.join(model_path))
    
    # Extract the features
    #gsr_signal.reshape((1, gsr_signal.shape[0]))
    train_outputs, latent_variable = model(torch.FloatTensor(gsrdata))
    return latent_variable.detach().numpy()[0];
    
    ############################ Feature Extraction Part ############################
    #################################################################################
    
