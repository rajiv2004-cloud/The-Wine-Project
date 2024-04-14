import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = {
    "Title": ["The Guv'nor, Spain", "Bread & Butter 'Winemaker's Selection' Chardon...",
              "Oyster Bay Sauvignon Blanc 2022, Marlborough", "Louis Latour Mâcon-Lugny 2021/22",
              "Bread & Butter 'Winemaker's Selection' Pinot N..."],
    "Description": ["We asked some of our most prized winemakers wo...", 
                    "This really does what it says on the tin. It’s...",
                    "Oyster Bay has been an award-winning gold-stan...",
                    "We’ve sold this wine for thirty years – and fo...",
                    "Bread & Butter is that thing that you can coun..."],
    "Price": [9.99, 15.99, 12.49, 17.99, 15.99],
    "Capacity": ["75CL", "75CL", "75CL", "75CL", "75CL"],
    "Region": [None, "California", "Marlborough", "Burgundy", "California"],
    "Style": ["Rich & Juicy", "Rich & Toasty", "Crisp & Zesty", "Ripe & Rounded", "Smooth & Mellow"],
    "Vintage": ["NV", "2021", "2022", "2022", "2021"],
    "Appellation": [None, "Napa Valley", None, "Macon", "Napa Valley"]
}

df = pd.DataFrame(data)

# Preprocess data
# Drop non-numeric columns for simplicity
df = df.drop(columns=["Title", "Description", "Capacity", "Region", "Style", "Appellation"])

# Encode categorical variables if any
le = LabelEncoder()
df["Vintage"] = le.fit_transform(df["Vintage"])

# Split data into features and target
X = df.drop(columns=["Price"])
y = df["Price"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predict prices for multiple samples
new_samples = pd.DataFrame({
    "Vintage": le.transform(["2021", "2022", "NV", "2022", "2021"])  # Wrap the values in a list
    # Add other features accordingly
})

# Add multiple predictions to the dataframe
new_samples["Predicted Price"] = model.predict(new_samples)
#Dataset elements
print(df)


# Plot actual vs. predicted prices for multiple samples
plt.plot(y_test.values, label='Actual Price', color='blue', marker='o')
plt.plot(new_samples["Predicted Price"].values, label='Predicted Price', color='red', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Prices')
plt.legend()
plt.show()
