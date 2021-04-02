# Big5-Personality-React-Flask
It's a project on which we can build a React app and call endpoints to make predictions. The data is downloaded originally from Kaggle of the big5 personalities "IPIP-FFM-data-8Nov2018"


## Technologies
- Back End
  - Flask
- Front End
  - React


# Revised Preparation
Create-react-app to create a basic React app to begin with. Next, bootstrap is loaded which allows us to create responsive websites for each screen size. In the App.js file a form with dropdowns and Predict and Reset Prediction buttons are added. Each form property is added to state and on pressing the Predict button, the data is sent to the Flask backend. The App.css file to add style to the page.

The Flask app has a POST endpoint /prediction. It accepts the input values as a json, converts it into an array and make prediction using the models stored in static folder as pickled files and return the prediction result as json.

Clone the repo to your computer and go inside it and open two terminals here.

In the first terminal, go inside the ui folder using cd ui. Do

    ```bash
      $ npm install
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



