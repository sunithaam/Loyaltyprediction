from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
import joblib
df=joblib.load('df.pkl')
app=Flask(__name__)
data=df
with open('model.pkl','rb') as model_file:
    rf_model=pickle.load(model_file)
@app.route('/')
def home():
    return render_template('first.html')

@app.route('/predict', methods=['POST'])
def predict():
     
    Items_Purchased=float(request.form['Items Purchased'])
    Total_Spent=float(request.form['Total Spent'])
    Satisfaction_Score=float(request.form['Satisfaction Score'])
    Warranty_Extension=float(request.form['Warranty Extension'])
    Revenue=float(request.form['Revenue'])
    Payment_Method_Cash=float(request.form['Payment Method_cash'])
    
    input_features=np.array([[Items_Purchased,Total_Spent,Satisfaction_Score,Warranty_Extension,Revenue,Payment_Method_Cash]])
    data=input_features
    pred=rf_model.predict(data)
    
    return render_template('first.html',pred_result=pred)
if __name__=='__main__':
    app.run(debug=True)