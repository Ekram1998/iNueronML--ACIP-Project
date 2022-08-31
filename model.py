import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the csv file
df = pd.read_csv("adult.csv")

# Level Encoding
for col in df.columns:
  if df[col].dtype=='object':
    le=LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

print(df.head())

# Select independent and dependent variable
target_name = 'salary'
# Separate object for targat feature
y = df[target_name]
# Separate object for input feature
X = df.drop(target_name,axis=1)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantian the mdoel
classifier = RandomForestClassifier()

# fit the model
classifier.fit(X_train,y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model_Rant.pkl", "wb"))
