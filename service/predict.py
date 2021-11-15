import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle, json, argparse, os, sys, torch.utils.data
from math import pi
from model_torch import Sentiment
from utils import paragraph_to_words, convert_and_pad
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Predictor():
    def __init__(self):        
        self.traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
        self.models = {}        
        self.load_models() 
        self.load_LSTM('cache/pytorch/')       
        
    
    def load_models(self):    
        for trait in self.traits:
            with open('cache/' + trait + '_model.pkl', 'rb') as f:
                self.models[trait] = pickle.load(f)


    def predict(self, X, traits='All', predictions='All'):
        predictions = {}
        if traits == 'All':
            for trait in self.traits:
                pkl_model = self.models[trait]                
                trait_scores = pkl_model.predict(X, regression=True).reshape(1, -1)
                # scaler = MinMaxScaler(feature_range=(0, 50))
                # print(scaler.fit_transform(trait_scores))
                # scaled_trait_scores = scaler.fit_transform(trait_scores)
                predictions['pred_s'+trait] = round(trait_scores.flatten()[0], 2)
                # predictions['pred_s'+trait] = scaled_trait_scores.flatten()
                trait_categories = pkl_model.predict(X, regression=False)
                predictions['pred_c'+trait] = str(trait_categories[0])
                # predictions['pred_c'+trait] = trait_categories
                trait_categories_probs = pkl_model.predict_proba(X)
                predictions['pred_prob_c'+trait] = round(trait_categories_probs[:, 1][0], 2)
                # predictions['pred_prob_c'+trait] = trait_categories_probs[:, 1]
        return predictions

    
    # Radar plot for personality
    def create_plot(self, predictions):        
        # Set data
        listofval=["Sample", predictions.get('pred_sOPN'), predictions.get('pred_sCON'), predictions.get('pred_sEXT'),
                    predictions.get('pred_sAGR'), predictions.get('pred_sNEU')]
        xdf = pd.DataFrame([listofval],columns=["Label","Openess","Conscientiousness","Extraversion","Agreeableness","Neuroticism"])
        # number of variable
        categories=list(xdf)[1:]
        N = len(categories)
        # We are going to plot the first line of the data frame.
        # But we need to repeat the first value to close the circular graph:
        values = xdf.loc[0].drop('Label').values.flatten().tolist()
        values += values[:1]
        values
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)        
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='grey', size=8)
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([1,2,3,4,5], ["1","2","3","4","5"], color="grey", size=7)
        plt.ylim(0,5)
        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        # Fill area
        ax.fill(angles, values, 'b', alpha=0.1)
        return plt #plt.show()
        

    def load_LSTM(self, model_dir):
        """Load the PyTorch model from the `model_dir` directory."""
        print("Loading model.")
        
        # Determine the device and construct the model.
        device = torch.device("cpu")
        modelLSTM = Sentiment(32, 100, 5000).to(device)
        # Load the store model parameters.
        model_path = os.path.join(model_dir, 'saved_weights.pt')
        with open(model_path, 'rb') as f:
            modelLSTM.load_state_dict(torch.load(f))
        # Load the saved word_dict.
        word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
        with open(word_dict_path, 'rb') as f:
            modelLSTM.word_dict = pickle.load(f)
        modelLSTM.to(device).eval()
        print("Done loading model.")
        self.modelLSTM = modelLSTM


    def deserialize_input(self, serialized_input_data, content_type):
        print('Deserializing the input data.')
        if content_type == 'text/plain':
            data = serialized_input_data.decode('utf-8')
            return data
        raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


    def serialize_output(self, prediction_output, accept):
        print('Serializing the generated output.')
        return str(prediction_output)


    def predict_LSTM(self, input_data, model):
        print('Inferring sentiment of input data.')
        device = torch.device("cpu")    
        if model.word_dict is None:
            raise Exception('Model has not been loaded properly, no word_dict.')        
        # data_X   - A sequence of length 500 which represents the converted review
        # data_len - The length of the review    
        data_X = None
        data_len = None    
        words = paragraph_to_words(input_data)
        data_X, data_len = convert_and_pad(model.word_dict, words)
        # Using data_X and data_len we construct an appropriate input tensor.
        # that our model expects input data of the form 'len, review[500]'.
        data_pack = np.hstack((data_len, data_X))
        data_pack = data_pack.reshape(1, -1)    
        data = torch.from_numpy(data_pack)
        data = data.to(device)
        # Make sure to put the model into evaluation mode
        model.eval()
        # Compute the result of applying the model to the input data. The variable `result` should
        # be a numpy array which contains a single integer which is either 1 or 0
        result = None    
        with torch.no_grad():
            out = model.forward(data)        
        result = out.numpy()        
        return result


if __name__ == '__main__':
    P = Predictor() 
    
    