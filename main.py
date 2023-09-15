from IPython.display import display
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.tree import  DecisionTreeClassifier

df = pd.read_csv("water_potability.csv")

# Cleaning Data From Null and duplicated Values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)



phMean_0 = df[df['Potability'] == 0]['ph'].mean(skipna=True)
df.loc[(df['Potability'] == 0) & (df['ph'].isna()), 'ph'] = phMean_0
phMean_1 = df[df['Potability'] == 1]['ph'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['ph'].isna()), 'ph'] = phMean_1
##################################### Imputing 'Sulfate' value #####################################
SulfateMean_0 = df[df['Potability'] == 0]['Sulfate'].mean(skipna=True)
df.loc[(df['Potability'] == 0) & (df['Sulfate'].isna()), 'Sulfate'] = SulfateMean_0
SulfateMean_1 = df[df['Potability'] == 1]['Sulfate'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['Sulfate'].isna()), 'Sulfate'] = SulfateMean_1
################################ Imputing 'Trihalomethanes' value #####################################
TrihalomethanesMean_0 = df[df['Potability'] == 0]['Trihalomethanes'].mean(skipna=True)
df.loc[(df['Potability'] == 0) & (df['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_0
TrihalomethanesMean_1 = df[df['Potability'] == 1]['Trihalomethanes'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_1





display(df.shape)
display(df.head())

x = df.drop(["Potability"], axis=1)
y = df["Potability"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=100)

tree = DecisionTreeClassifier(criterion='entropy',random_state=0)
tree.fit(x_train, y_train)
prediction = tree.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, prediction))
