from flask import Flask, render_template
import pandas as pd
import joblib

def create_lagged_features(df, n_lags=10):
    lagged_df = df.copy()
    for lag in range(1, n_lags + 1):
        lagged_df[f'lag_{lag}'] = df['ortalama_fiyat'].shift(lag)
    return lagged_df

def get_last_day():
    # Load the dataset
    dataset = pd.read_csv("dataset/son_hali_guncel.csv")  # Correct path to your dataset
    dataset['tarih'] = pd.to_datetime(dataset['tarih'])

    # Find the latest date for each product
    latest_dates_per_product = dataset.groupby('urun_ad')['tarih'].max().reset_index()

    # Add the latest dates to the main dataset
    dataset_with_latest_dates = dataset.merge(latest_dates_per_product, on='urun_ad', suffixes=('', '_latest'))

    # Filter for the records just before the latest date for each product
    dataset_filtered = dataset_with_latest_dates[dataset_with_latest_dates['tarih'] < dataset_with_latest_dates['tarih_latest']]



    # Split the date into year/month/day
    dataset_filtered['year'] = dataset_filtered['tarih'].dt.year
    dataset_filtered['month'] = dataset_filtered['tarih'].dt.month
    dataset_filtered['day'] = dataset_filtered['tarih'].dt.day

    # Drop unnecessary columns including 'tarih' and 'tarih_latest'
    dataset_filtered = dataset_filtered.drop(['tarih', 'tarih_latest', 'birim', '_id'], axis=1)

    # One-hot encoding
    dataset_encoded = pd.get_dummies(dataset_filtered, columns=['urun_ad'])

    # Create lagged features
    lagged_data = create_lagged_features(dataset_encoded)
    lagged_data = lagged_data.iloc[10:].fillna(0)  # Skip the first 10 rows and fill missing values

    # Prepare the features for prediction
    X = lagged_data.drop('ortalama_fiyat', axis=1)
    print(X)

    # Load the model
    rf_model = joblib.load("model/random_forest_model_v4_all_2019.joblib")  # Correct path to your model

    # Make predictions
    rf_predictions = rf_model.predict(X)

    # Re-load the dataset to get the actual values and relevant date
    dataset_latest = dataset_with_latest_dates[dataset_with_latest_dates['tarih'] == dataset_with_latest_dates['tarih_latest']]
    relevant_date = dataset_latest['tarih'].dt.strftime('%Y-%m-%d').iloc[0]

    # Creating dictionaries for actual values and predictions
    actual_values_dict = {product: actual for product, actual in zip(dataset_latest['urun_ad'], dataset_latest['ortalama_fiyat'])}
    predictions_dict = {product: prediction for product, prediction in zip(dataset_latest['urun_ad'], rf_predictions)}

    return actual_values_dict, predictions_dict, relevant_date


actual_values_dict, predictions_dict, relevant_date = get_last_day()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/meyve')
def meyve():
    return render_template('araclar.html', actual_values_dict=actual_values_dict, predictions_dict=predictions_dict, relevant_date=relevant_date)

@app.route('/sebze')
def sebze():
    return render_template('araclar_2.html', actual_values_dict=actual_values_dict, predictions_dict=predictions_dict, relevant_date=relevant_date)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")