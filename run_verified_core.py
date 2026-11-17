"""Reproduce the slide-safe core benchmark from raw January-2024 CSVs.

This runner intentionally excludes temporal/OOD results until those protocols are rerun
from real predictions. It is the authoritative reproducibility entry point for the verified
core model table used in the final presentation.
"""
from pathlib import Path
import sys, re, json
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'src'))
from final_core_experiments import load_data, split_data, SKILLS

OUT = BASE / 'results' / 'final_core'
OUT.mkdir(parents=True, exist_ok=True)

def score(name, y, p, note):
    return {
        'model': name,
        'accuracy': accuracy_score(y,p),
        'macro_precision': precision_score(y,p,average='macro',zero_division=0),
        'macro_recall': recall_score(y,p,average='macro',zero_division=0),
        'macro_f1': f1_score(y,p,average='macro',zero_division=0),
        'note': note,
    }

print('Preparing verified data...')
df = load_data()
df, tr, va, te = split_data(df)
train, val, test = df.iloc[tr], df.iloc[va], df.iloc[te]
ytr, yt = train.target_role, test.target_role
rows=[]

legacy_cols=['legacy__'+s for s in SKILLS]
strict_cols=['strict__'+s for s in SKILLS]

# Validation-selected hyperparameters from the verified tuning artifacts.
m=LogisticRegression(C=.25,class_weight='balanced',max_iter=3000,random_state=42).fit(train[legacy_cols],ytr)
rows.append(score('Legacy skills LR',yt,m.predict(test[legacy_cols]),'Provided-skill reproducibility baseline'))

m=LogisticRegression(C=1.0,class_weight='balanced',max_iter=3000,random_state=42).fit(train[strict_cols],ytr)
rows.append(score('Strict masked-description skills LR',yt,m.predict(test[strict_cols]),'40 skills from masked description'))

m=DecisionTreeClassifier(max_depth=8,min_samples_leaf=5,class_weight='balanced',random_state=42).fit(train[strict_cols],ytr)
rows.append(score('Strict skills Decision Tree',yt,m.predict(test[strict_cols]),'Interpretable nonlinear baseline'))

for col,c,name,note in [
    ('text_unmasked',1.0,'TF-IDF LinearSVC unmasked','Leakage diagnostic only; not eligible final model'),
    ('text_masked',.25,'TF-IDF LinearSVC masked','Selected leakage-controlled text model'),
]:
    v=TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=2,sublinear_tf=True,strip_accents='unicode')
    Xtr=v.fit_transform(train[col]); Xte=v.transform(test[col])
    m=LinearSVC(C=c,class_weight='balanced',random_state=42).fit(Xtr,ytr)
    rows.append(score(name,yt,m.predict(Xte),note))

# Hybrid uses the same masked TF-IDF representation plus strict 40-skill indicators.
v=TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=2,sublinear_tf=True,strip_accents='unicode')
Xtr_t=v.fit_transform(train.text_masked); Xte_t=v.transform(test.text_masked)
Xtr=sp.hstack([Xtr_t, sp.csr_matrix(train[strict_cols].values)],format='csr')
Xte=sp.hstack([Xte_t, sp.csr_matrix(test[strict_cols].values)],format='csr')
m=LinearSVC(C=1.0,class_weight='balanced',random_state=42,max_iter=5000).fit(Xtr,ytr)
rows.append(score('Hybrid masked TF-IDF + strict skills LinearSVC',yt,m.predict(Xte),'Hybrid did not outperform masked text-only'))

res=pd.DataFrame(rows).sort_values('macro_f1',ascending=False)
res.to_csv(OUT/'model_comparison_rerun.csv',index=False)
print(res[['model','accuracy','macro_f1']].to_string(index=False))
print('\nSaved:', OUT/'model_comparison_rerun.csv')
