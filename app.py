from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    # Handle the case when the model file is not found
    print("Model file not found.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input data from the request form
        core_board_3A = float(request.form['coreboard3A'])
        core_board_7A = float(request.form['coreboard7A'])
        
        # Make a prediction using the loaded model
        composition_to_predict = pd.DataFrame({'3A': [core_board_3A], '7A': [core_board_7A]})
        predicted_strength = model.predict(composition_to_predict)
        
        # Return the prediction result as a response
        response = {
            "predicted_strength": predicted_strength.tolist()
        }
        
        # Pass the prediction result to the index.html template
        return render_template('index.html', prediction_result=response["predicted_strength"])
    except Exception as e:
        # Handle exceptions and return an error response
        return render_template('index.html', prediction_result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
