```python
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


# =========================================================
# 🌱 Crop Recommendation Model
# =========================================================
def get_crop_model():

    print("Loading crop dataset...")
    df = pd.read_csv("data/Crop_recommendation.csv")

    X = df.drop("label", axis=1)
    y = df["label"]

    print("Training crop model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    return model


# =========================================================
# 🌿 Fertilizer Recommendation Model
# =========================================================
def get_fertilizer_model():

    print("Loading fertilizer dataset...")
    df = pd.read_csv("data/Fertilizer_Prediction.csv")

    df = df.rename({
        'Fertilizer Name': 'Fertilizer',
        'Crop Type': 'Crop_Type',
        'Soil Type': 'Soil_Type'
    }, axis=1)

    target = "Fertilizer"

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    print("Training fertilizer model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# =========================================================
# 🔢 Input Converter Function
# =========================================================
def get_input(x):

    x_structure = {
        "Temparature": 0, "Humidity": 1, "Moisture": 2, "Nitrogen": 3,
        "Potassium": 4, "Phosphorous": 5,
        "Black": 6, "Clayey": 7, "Loamy": 8, "Red": 9, "Sandy": 10,
        "Barley": 11, "Cotton": 12, "Ground Nuts": 13, "Maize": 14,
        "Millets": 15, "Oil seeds": 16, "Paddy": 17, "Pulses": 18,
        "Sugarcane": 19, "Tobacco": 20, "Wheat": 21
    }

    output = np.zeros(len(x_structure))

    output[0] = x[0]
    output[1] = x[1]
    output[2] = x[2]
    output[3] = x[3]
    output[4] = x[4]
    output[5] = x[5]

    if x[6] in x_structure:
        output[x_structure[x[6]]] = 1

    if x[7] in x_structure:
        output[x_structure[x[7]]] = 1

    return output
```
