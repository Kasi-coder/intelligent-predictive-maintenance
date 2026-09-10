"""Train the predictive-maintenance model from a CSV."""
from pathlib import Path
import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score
from src.preprocessing import make_features, IQRClipper

ROOT=Path(__file__).resolve().parents[1]
CSV=ROOT/'data'/'ai4i2020_reproducible.csv'
Xraw=pd.read_csv(CSV)
y=Xraw['Machine failure']
X=make_features(Xraw)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
num=[c for c in X.columns if X[c].dtype!='object']; cat=['Type']
pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
model=Pipeline([('outlier_clipper',IQRClipper()),('preprocess',pre),('model',RandomForestClassifier(n_estimators=400,max_depth=10,min_samples_leaf=2,class_weight='balanced_subsample',random_state=42,n_jobs=-1))])
model.fit(Xtr,ytr)
p=model.predict_proba(Xte)[:,1]
print(classification_report(yte,(p>=.5).astype(int),digits=4))
print('ROC-AUC:',roc_auc_score(yte,p)); print('PR-AUC:',average_precision_score(yte,p))
joblib.dump(model,ROOT/'models'/'predictive_maintenance_model.pkl')
print('Saved model.')
