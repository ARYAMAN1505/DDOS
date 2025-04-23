from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load pre-trained model, scaler, and label encoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('best_rf.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = "Please enter necessary values"
    if request.method == 'POST':
        try:
            serror_rate = float(request.form['serror_rate'])
            srv_serror_rate = float(request.form['srv_serror_rate'])
            same_srv_rate = float(request.form['same_srv_rate'])
            dst_host_serror_rate = float(request.form['dst_host_serror_rate'])
            dst_host_srv_serror_rate = float(request.form['dst_host_srv_serror_rate'])

            input_data = [serror_rate, srv_serror_rate, same_srv_rate, dst_host_serror_rate, dst_host_srv_serror_rate]
            input_scaled = scaler.transform([input_data])
            result = model.predict(input_scaled)
            prediction = label_encoder.inverse_transform(result)[0]
        except Exception as e:
            prediction = "Error in prediction. Please check input values."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

