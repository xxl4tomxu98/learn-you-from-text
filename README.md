# React-Flask Text Analysis for Personality and Sentiment
The models for personality prediction are Random Forest Regressor and Random Forest Classifier. The models are trained on a dataset from the myPersonality project (https://sites.google.com/michalkosinski.com/mypersonality). Similar dataset exists regarding big 5 personality test (https://openpsychometrics.org/tests/IPIP-BFFM/). Models produce a predicted personality score, using the regression model, and a probability of the binary class, using the classification model, for each personality trait.

The model for sentiment analysis is a pytorch recurrent neural network LSTM deep learning model that is trained using the IMDb dataset (http://ai.stanford.edu/~amaas/data/sentiment/). There are multiple ways to download this large review dataset. Simplist would be use simple shell script:

``` shell
  %mkdir ../data
  !wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
  !tar -zxf ../data/aclImdb_v1.tar.gz -C ../data
```

The data files are all stored for evaluations and checking in the data folder.

## Technologies
- Back End
  - Flask
- Front End
  - React


# Web App Deployment Using React.js
Create-react-app to create a basic React app to begin with. Next, bootstrap is loaded which allows us to create responsive websites for each screen size. In the App.js file a form with textarea and Predict buttons are added. Each form property is added to state and on pressing the Predict button, the data is sent to the Flask backend. The App.css file to add style to the page.

The Flask app has a POST endpoint /prediction. It accepts the input values as a json, converts it into an array and make prediction using the models stored in static folder as pickled files and return the prediction result as json.

Clone the repo to your computer and go inside it and open two terminals here.

In the first terminal, go inside the ui folder using cd ui. Do

    ```bash
      $ npm install
    ```
To see the UI in a development mode, simply run:

    ``` bash
      npm install
      npm start
    ```

To run the UI on server, we will use serve. We will begin by installing the serve globally, post which, we’ll build our application and then finally run the UI using serve on port 3000.

    ```bash
      npm install -g serve
      npm run build
      serve -s build -l 3000
    ```
You can now go to localhost:3000 to see that the UI is up and running. But it won’t interact with the Flask service which is still not up. So, Preparing the service on the second terminal, move inside the service folder using cd service. We begin by creating a virtual environment and install dependencies using conda install from conda miniforge3 libray and Python 3.9.2. Finally, we’ll run the Flask app.

    ```bash
      conda activate ml
      python app.py      
    ```
This will start up the service on 127.0.0.1:5000 backend. One can go to the localhost to test the backend via the server.



