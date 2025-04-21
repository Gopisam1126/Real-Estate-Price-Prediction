import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('bengaluru_house_prices.csv')

scalar = MinMaxScaler()
model = RandomForestRegressor()

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def process_sqft(value):
    value_str = str(value)
    if '-' in value_str:
        try:
            lower_str, upper_str = [part.strip() for part in value_str.split('-')]
            lower_val = float(lower_str)
            upper_val = float(upper_str)
            return (lower_val + upper_val) / 2
        except ValueError:
            return np.nan
    else:
        try:
            return float(value_str)
        except ValueError:
            return np.nan

def process_size(value):
    value_str = str(value)
    try:
        lower_val, _ = [part.strip() for part in value_str.split(' ')]
        return float(lower_val)
    except ValueError:
        return np.nan

def remove_outlier_pps(df):
    df1 = pd.DataFrame()
    for _, sub_df in df.groupby('location'):
        m = np.mean(sub_df['price_per_sqft'])
        std = np.std(sub_df['price_per_sqft'])
        new_df = sub_df[(sub_df['price_per_sqft'] > (m - std)) & (sub_df['price_per_sqft'] <= (m + std))]
        df1 = pd.concat([df1, new_df], ignore_index=True)
    return df1

# Preprocessing steps
df['balcony'] = df['balcony'].fillna(df['balcony'].median())
df['bath'] = df['bath'].fillna(df['bath'].median())
df['total_sqft'] = df['total_sqft'].apply(process_sqft)
df['size'] = df['size'].apply(process_size)
df.drop('society', axis=1, inplace=True)
df = df.dropna()

df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']
df = remove_outlier_pps(df)

df = df[df['bath'] <= (df['size'] + 2)]

df['location'] = df['location'].apply(lambda x: x.strip())
loc_stat = df['location'].value_counts(ascending=False)
loc_less_ten = loc_stat[loc_stat <= 10]
df['location'] = df['location'].apply(lambda x: 'other' if x in loc_less_ten else x)
df = df[~(df['total_sqft'] / df['size'] < 300)]

df_n = pd.get_dummies(df, drop_first=True, columns=['area_type', 'location'])

x = df_n.drop(['price', 'availability'], axis='columns')
y = df_n['price']

x_numeric = x.select_dtypes(include=[np.number])
X_scaled = scalar.fit_transform(x_numeric)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=100)
model.fit(X_train, y_train)
cross_val_sc = cross_val_score(RandomForestRegressor(), X_scaled, y, cv=10)

def predict_price(location, sqft, bath, size):
    input_features = {col: 0 for col in x.columns}

    if 'total_sqft' in input_features:
        input_features['total_sqft'] = sqft
    if 'bath' in input_features:
        input_features['bath'] = bath
    if 'size' in input_features:
        input_features['size'] = size

    loc_col = f'location_{location}'
    if loc_col not in input_features:
        print(f"Warning: Location '{location}' not found. Using default 'other' category.")
        loc_col = 'location_other'
        if loc_col not in input_features:
            raise ValueError("Neither the input location nor 'other' are in the training features.")

    input_features[loc_col] = 1

    input_df = pd.DataFrame([input_features])
    input_numeric = input_df[x_numeric.columns]

    input_scaled = scalar.transform(input_numeric)

    return model.predict(input_scaled)[0]

predicted_val = predict_price('9th Phase JP Nagar', 2000, 2, 2)
print(f"Predicted price: {predicted_val:.4f}")

print(f"Cross Validation Score : {cross_val_sc.mean()}")
print(f"Stand alone score : {model.score(X_test, y_test)}")

feature_list = x_numeric.columns.tolist()

dict1={'model': model, 'scaler': scalar, 'columns': feature_list}
with open("house_prices_model.pkl",'wb') as obj1:
  pickle.dump(dict1,obj1)
with open("house_prices_model.pkl",'rb') as obj2:
  var1=pickle.load(obj2)