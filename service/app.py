from flask import Flask, render_template, request, jsonify
from model import Model
from predict import Predictor


app = Flask(__name__, static_folder='../ui/build', static_url_path='/')
M = Model()
predictor = Predictor()


class imgcounterclass():
    """Class container for indexing OCEAN plots."""

    _counter = 0

    def addcounter(self):
        self._counter += 1


image_counter=imgcounterclass()


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def react_root(path):
    print("path", path)
    if path == 'favicon.ico':
        return app.send_static_file('favicon.ico')
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.json
    prediction =  predictor.predict([text])
    # prediction = pd.DataFrame(prediction).to_html()
    # return prediction
    # return jsonify({'prediction': str(prediction)}) 
    image_counter.addcounter()   
    predictor.create_plot(prediction).savefig(f'../ui/public/prediction{image_counter._counter}.png',
                                                bbox_inches="tight")
    return jsonify({"prediction": prediction, "url":image_counter._counter})
    #
    # return render_template('index.txt', predictions=prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
