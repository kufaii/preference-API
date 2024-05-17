from fastapi import FastAPI, Request
import pickle
import uvicorn
import pandas as pd


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Your API is UP!"}

# Check model
@app.get('/check-model')
def check_model():
    try:
        with open('models/predict_preference.pkl', 'rb') as model:
            model = pickle.load(model)
        result = {
            'status': 'OK',
            'message': 'Model is ready to use'
        }
        return result
    except Exception as e:
        result = {
            'status': 'Error',
            'message': str(e)
        }
        return result
    
# Predict
@app.post('/predict')
async def predict(request: Request):
    # get data from request
    data = await request.json()
    inf = pd.DataFrame(data, index=[0])

    # store to variable
    q1 = data['q1']
    q2 = data['q2']
    q3 = data['q3']
    q4 = data['q4']
    q5 = data['q5']
    q6 = data['q6']
    q7 = data['q7']
    
    
    # load model
    with open('models/predict_preference.pkl', 'rb') as model:
        model = pickle.load(model)
    
    # Predict
    try:
        prediction = model.predict(inf)
        result = {
            'status': 'OK',
            'message': 'Prediction is ready',
            'prediction': prediction[0]
        }
        return result
    except Exception as e:
        result = {
            'status': 'Error',
            'message': str(e)
        }
        return result
    
# Run API
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)