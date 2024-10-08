import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from helpers import (convert_to_float, export_dataframe, data_analysis, plot_variances,
                     proportion_nan, count_three_decimals_values)

df = pd.read_csv('LinearRegression.csv', sep=';', decimal=',')
df = df.map(convert_to_float)  # convert data to float if is necessary

# Convert data to numpy array
numpy_df = df.to_numpy()

# Scale the data

scaler = StandardScaler()
numpy_df_scaled = scaler.fit_transform(numpy_df)  # scale data to mean 0 and variance 1

# Impute missing values with Linear Regression
for i in range (numpy_df_scaled.shape[1]):
    column = numpy_df_scaled[:, i]
    missing_mask = np.isnan(column)
    if np.any(missing_mask):
        not_missing_mask = ~missing_mask
        x_train = np.where(not_missing_mask)[0].reshape(-1, 1)
        y_train = column[not_missing_mask]
        x_test = np.where(missing_mask)[0].reshape(-1, 1)

        model = LinearRegression()
        model.fit(x_train, y_train)
        column[missing_mask] = model.predict(x_test)
        numpy_df_scaled[:, i] = column

# Invert Scaling

numpy_df_imputed_scaled_back = scaler.inverse_transform(numpy_df_scaled)
df_imputed_scaled_back = pd.DataFrame(numpy_df_imputed_scaled_back, columns=df.columns)
df_imputed_scaled_back = df_imputed_scaled_back.map(lambda x: round(x, 3))

# Export Imputed DataFrame to CSV

export_dataframe(df_imputed_scaled_back, 'LinearRegression_imputed')  # explore imputed data to CSV file

# Analyze and test the imputed data

data_analysis(df_imputed_scaled_back)
