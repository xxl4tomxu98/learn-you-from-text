import pandas as pd
import pickle
from math import pi
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Predictor():
    def __init__(self):        
        self.traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
        self.models = {}
        self.load_models()        
        
    
    def load_models(self):
    
        for trait in self.traits:
            with open('static/' + trait + '_model.pkl', 'rb') as f:
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
        

if __name__ == '__main__':
    P = Predictor() 
    
    