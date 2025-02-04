##  Insurance Prediction API(FastAPI)
This project provides an API for predicting insurance charges based on user input using a trained machine learning model 

##  Table of Contents
- [Description](...)
- [Requirements](...)
- [Getting Started](...)
- [1. Train and Save the Model](...)
- [2. Deploy FastAPI](...)
- [3. Run Streamlit](...)
- [Usage](...)
- [Endpoints](#endpoints)
- [Example Input and Output](...)
- [File Structure](...)
- [License](#lisence) 

##  Description


This FASTAPI application provides an API and a Streamlit application for predicting insurance charges based on users input. The model is trained on the following features:

- Age
- Gender
- BMI
- Number of children
- Smoker status
- Region


The API predicts insurance charges in usd based on these features.

##  Requirements
To set up and run this project, you’ll need the following Python packages:

- `fastapi`
- `uvicorn`
- `scikit-learn`
- `pandas`
- `joblib`
- `numpy`
- `xgboost`

You can install these dependencies by running:

```bash
pip install -r requirements.txt
```

## Getting Started
Follow these steps to set up and run the project.

1. Train and Save Model

  Train the XGBoost regression model using Scikit-Learn and XGBoost regressor, save the trained model to files for deployment:
  ```bash
  python model_build.py
  ```
This will create the model.pkl file in the model/ directory.

2. The FastAPI application (`api.py`) loads the saved model and provide an endpoint for prediction
  ```bash
  uvicorn api:app --reload
  ```
This will start the FastAPI server at http://127.0.0.1:8001


Run Streamlit The Streamlit app allows users to input values and retrieve predictions from the FastAPI server. To start Streamlit, run:
streamlit run app.py
The Streamlit app will open in a browser window at http://localhost:8501.

##Endpoints
![API Image](![api_figure.png](src%2Fapi_figure.png))

POST /api/predict
-Description: Accepts insurance features value and returns a predicted insurance values in dollars
-Input JSON:
```bash
{
  "age": 30,
  "gender": male,
  "bmi": 27,
  "children": 4,
  "smoker": no,
  "region": "southwest"
}
-Output JSON:
```bash
{
  "predicted_charges": 4,5000.78
}
Streamlit Application
The Streamlit app provides an interface for users to input feature values and get predictions. When the Predict button is clicked, the app sends the data to the FastAPI server and displays whether the tumor is benign or malignant.

Example Input and Output
Example Input:

Age= 45
Gender = Female
BMI = 28.7
Number of children = 2
Smoker status = yes
Region = Southeast

Example Output:

Predicted charges: $17,900.55

File Structure
The project directory is structured as follows:

📦 XGBoost_regressor API
├─ data
│  └─ data.csv
├─ model
│  ├─ model.pkl
├─ src
├─ .gitignore
├─ app.py
├─ api.py
├─ README.md
└─ requirements.txt

##License
This project is licensed under [![License: MIT](...)](...)