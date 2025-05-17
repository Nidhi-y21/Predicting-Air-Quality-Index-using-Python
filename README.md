# Predicting-Air-Quality-Index-using-Python
Let us see how to predict the air quality index using Python. AQI is calculated based on chemical pollutant quantity. By using machine learning, we can predict the AQI.

AQI: The air quality index is an index for reporting air quality on a daily basis.  In other words, it is a measure of how air pollution affects one's health within a short time period. The AQI is calculated based on the average concentration of a particular pollutant measured over a standard time interval. Generally, the time interval is 24 hours for most pollutants, and 8 hours for carbon monoxide and ozone.

We can see how air pollution is by looking at the AQI

AQI Level	AQI Range
Good	0 - 50
Moderate	51 - 100
Unhealthy	101 - 150
Unhealthy for Strong People	151 - 200
Hazardous	201+
Let's find the AQI based on Chemical pollutants using Machine Learning Concept. 

Data Set Description
It contains 7 attributes, of which 6 are chemical pollution quantities and one is Air Quality Index. AQI Value, CO AQI Value, Ozone AQI Value, NO2 AQI Value, PM2.5 AQI Value, lat,LNG are independent attributes. air_quality_index is a dependent attribute. Since air_quality_index is calculated based on the 7 attributes.

As the data is numeric and there are no missing values in the data, so no preprocessing is required. Our goal is to predict the AQI, so this task is either Classification or regression. So as our class label is continuous, regression technique is required.

Regression is supervised learning technique that fits the data in a given range. Example Regression techniques in Python:

Random Forest Regressor
Ada Boost Regressor
Bagging Regressor
Linear Regression etc.
# importing pandas module for data frame
import pandas as pd

# loading dataset and storing in train variable
train= pd.read_csv("/content/AQI and Lat Long of Countries.csv")
# display top 5 data
train.head()
 Output:




from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

#CREATING MODEL
m1= RandomForestRegressor()


train1= train.drop(['AQI Value'], axis=1)

target= train['AQI Value']

print(train1)
print(target)
#FITTING THE MODEL
m1.fit(train1, target)

m1.score(train1, target)*100

# predicting the model with other values (testing the data)
prediction_result= m1.predict([[1, 10, 5, 11, 10, 5]])

print(prediction_result)


# Adaboost model
# importing module

from sklearn.ensemble import AdaBoostRegressor


# defining model
m2 = AdaBoostRegressor()

# Fitting the model
m2.fit(train1, target)

'''AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear',
                  n_estimators=50, random_state=None)'''

m2.score(train1, target)*100

# predicting the model with other values (testing the data)
# so AQI is 48.73051389
m2.predict([[1, 45, 67, 34, 5, 23]])
Linear Regression:

Adaboost:



By this, we can say that by given test data we got 9.9 and 46.5952 so the air is healthy.
