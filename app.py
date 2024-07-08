import pickle

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder
from sklearn.compose import ColumnTransformer


app = Flask(__name__)

model = pickle.load(open('xgb_classifier.pkl','rb'))

@app.route("/",methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def prediction():
    if request.method == 'POST':
        age = float(request.form.get("age"))
        height = float(request.form.get("height"))
        weight = float(request.form.get("weight"))
        ap_hi = float(request.form.get("ap_hi"))
        ap_lo = float(request.form.get("ap_lo"))
        gender = float(request.form.get("gender"))
        cholesterol = float(request.form.get("Cholesterol"))
        glucose = float(request.form.get("Glucose"))
        smoke = float(request.form.get("Smoke"))
        alcohol = float(request.form.get("Alcohol"))
        active = float(request.form.get("Actve"))
        bmi = float(request.form.get("bmi"))
        bp_category = request.form.get("bp_category")

        bp_cat = [0,0,0]

        if bp_category=='Normal':
            bp_cat[2]=1
        elif bp_category=='Hypertension Stage 1':
            bp_cat[0]=1
        elif bp_category=='Hypertension Stage 2':
            bp_cat[1]=1

        inputs = [height,weight,ap_hi,ap_lo,age,bmi,gender,
                  cholesterol,glucose,smoke,alcohol,active]
        
        inputs.extend(bp_cat)
        inputs = np.array(inputs).reshape(1,-1)

        #ct = ColumnTransformer(transformers=[('scaler',RobustScaler(),[0,1,2,3,4,5])],
         #                      remainder='passthrough')
        #inputs_transformed = ct.fit(inputs)

        
        pred = model.predict(inputs)

        if pred[0]==0:
            prediction = 'No disease'
        else:
            prediction='Disease'

    return render_template("index.html",data=prediction)


if __name__=='__main__':
    app.run(debug=True)
   
     