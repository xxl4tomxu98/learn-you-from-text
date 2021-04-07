## Data Preprocessing
    Data are read in from local storage where they were downloaded. The big 5 dataset are vectorized by TFIDF vectorizer. It is in .csv format so it was simply imported into pandas dataframe. The IMDb dataset is in .txt format so it was imported in using beautifulsoup. It also needs a lot more cleaning and padding to allow pytorch model to use the data. I was loaded in two steps, first, it was preprocessed into a pickled preprocessed datafile and stored locally. Second, a word dictionary is built based on word appearing frequencies and word embedding is done based on the dictionary and stored in the "train.csv" file.

    ``` shell
        python data_prep.py
        python process_torch.py
    ```

## Model Fitting
    The big 5 prediction model is model.py and to fit the model one need to run:

    ``` shell
        python model.py
    ```

    in the service folder, this trains the models on the myPersonalty status data and creates five pickle files corresponding to each personality trait in the static folder.

    The sentiment model is called from process_torch.py file. The preprocessed and padded data is loaded into pytorch tensor dataset and sent to pytorch sentiment model for predictions. Quality of the model can be tested using the corresponding test set and accuracy of the model can be evaluated based on scores calculated.

    ``` shell
        python process_torch.py
    ```

## Model Evaluations
    models can be evaluated and compared by running:

    ``` shell
        python model_eval.py
    ```


## Predictions
    user text inputs will follow similar cleaning and padding sequence before they can be sent agianst the models for predictions. Prediction tests can be individually conducted by running:

    ``` shell
        python predict.py
    ```
    most of the time, the flask will be run to start the backend server by running the web app:

    ``` shell
        python app.py
    ```

    This runs the web app on the local environment. Visit localhost:5000 to view the web app.