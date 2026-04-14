import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.DataFrame({
    "views": [100, 200, 300],
    "likes": [10, 20, 30]
})

model = LinearRegression()
model.fit(data[["views"]], data["likes"])

print("Prediction for 400 views:", model.predict([[400]]))