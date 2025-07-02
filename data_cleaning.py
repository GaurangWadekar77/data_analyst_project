import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"c:\Users\Admin\Desktop\diabetes.csv")

df = df.head(70)
x = df.drop("Outcome",axis=1)
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test,predictions))