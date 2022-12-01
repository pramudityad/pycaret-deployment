import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
import os

ON_HEROKU = os.environ.get('ON_HEROKU')
# 2. Create the app object
app = FastAPI()

#. Load trained Pipeline
model = load_model('diamond-pipeline')

# Define predict function
@app.post('/predict')
def predict(carat_weight, cut, color, clarity, polish, symmetry, report):
    data = pd.DataFrame([[carat_weight, cut, color, clarity, polish, symmetry, report]])
    data.columns = ['Carat Weight', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry', 'Report']

    predictions = predict_model(model, data=data) 
    return {'prediction': int(predictions['Label'][0])}

if __name__ == '__main__':
    if ON_HEROKU:
        # get the heroku port
        port = int(os.environ.get('PORT', 37651))
    else:
        port = 8000
    uvicorn.run(app, host='127.0.0.1', port=port)