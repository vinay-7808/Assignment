(1) Project.ipynb:

In this notebook, I explored various machine learning models such as Logistic Regression, KNN, and Random Forest Classifier.
To determine the most accurate model, I employed a systematic approach. I started by importing the dataset and addressed missing data and outliers using the z-score method. Subsequently, I performed feature engineering, introducing two distinct features. Following this, I scaled the data, making it ready for model training.
After assessing the accuracy of different machine learning models, I identified Logistic Regression as the top performer. Therefore, I selected this model for the final predictions.


(2) Deployment:

For deployment, I utilized FastAPI, implementing two key files: app.py and logistic_regression_model.pkl.
The app.py file serves as the local API, facilitating the interaction between the user and the machine learning model. Users can run app.py, input data, and access predictions by visiting localhost:8000/host.
