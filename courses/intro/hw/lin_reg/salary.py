import pandas as pd
from sklearn.feature_extraction import text, DictVectorizer
import scipy
from sklearn.linear_model import Ridge

data = pd.read_csv("salary-train.csv")  # heavy, excluded from Git. Format of "salary-train-sample.csv" should be used
data["FullDescription"] = data["FullDescription"].apply(str.lower).replace('[^a-z0-9]', ' ', regex=True)
tf = text.TfidfVectorizer(min_df=5)
idfs = tf.fit_transform(data["FullDescription"])
data_test = pd.read_csv("salary-test-mini.csv")

data['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X = scipy.sparse.hstack((idfs, X_train_categ))
X_test = scipy.sparse.hstack((tf.transform(data_test.FullDescription), X_test_categ))
model = Ridge(alpha=1, random_state=241)
model.fit(X, data.SalaryNormalized)
print("\n".join(model.predict(X_test)))
