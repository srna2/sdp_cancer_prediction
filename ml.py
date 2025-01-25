import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import pickle

warnings.filterwarnings("ignore")

dataset = pd.read_csv("cancer.csv")

X = dataset[
    ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concave_points_mean',
     'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'concave_points_worst']]
Y = dataset['diagnosis']

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)  # "M" -> 1, "B" -> 0

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

rf_classifier_model = RandomForestClassifier(class_weight="balanced", random_state=2)
rf_classifier_model.fit(X_train, Y_train)

pickle.dump(rf_classifier_model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

