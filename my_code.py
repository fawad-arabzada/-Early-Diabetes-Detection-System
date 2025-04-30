import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_data():
    file_path = "diabetes1.csv"  
    return pd.read_csv(file_path)

def preprocess_data(df, scaler=None):
    features = ["Pregnancies", "BloodPressure", "SkinThickness", "Insulin", 
                "BMI", "Age", "Glucose", "DiabetesPedigreeFunction"]
    
    if scaler is None:
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
    else:
        df[features] = scaler.transform(df[features])
    
    return df, scaler

def get_train_test_split(df, target_column="Outcome"):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(solver='newton-cg', max_iter=1000)
    model.fit(X_train, y_train)
    return model

data = load_data()
data, scaler = preprocess_data(data)
X_train, X_test, y_train, y_test = get_train_test_split(data)
model = train_logistic_regression(X_train, y_train)

print("Model trained successfully!")
