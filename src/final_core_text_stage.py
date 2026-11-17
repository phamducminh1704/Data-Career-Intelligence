from pathlib import Path
import sys, time, re, json
import joblib, pandas as pd, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from final_core_experiments import load_data, split_data, SKILLS, ROLES, slug

BASE=Path(__file__).resolve().parents[1]; OUT=BASE/'results'/'final_core'; FIG=BASE/'figures'/'final_core'; MOD=BASE/'models'/'final_core'
for d in (OUT,FIG,MOD): d.mkdir(parents=True,exist_ok=True)

def metric(name,y,p,tr,inf,note):
    return {'model':name,'accuracy':accuracy_score(y,p),'macro_precision':precision_score(y,p,average='macro',zero_division=0),'macro_recall':recall_score(y,p,average='macro',zero_division=0),'macro_f1':f1_score(y,p,average='macro',zero_division=0),'train_seconds':tr,'test_inference_seconds':inf,'note':note}

def save_diag(name,y,p,test):
    s=slug(name); pd.DataFrame(classification_report(y,p,output_dict=True,zero_division=0)).T.to_csv(OUT/f'classification_report__{s}.csv'); cm=confusion_matrix(y,p,labels=ROLES); pd.DataFrame(cm,index=ROLES,columns=ROLES).to_csv(OUT/f'confusion_matrix__{s}.csv')
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm); ax.set_xticks(range(4),ROLES,rotation=28,ha='right'); ax.set_yticks(range(4),ROLES); ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(name)
    for i in range(4):
        for j in range(4): ax.text(j,i,int(cm[i,j]),ha='center',va='center')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04);fig.tight_layout();fig.savefig(FIG/f'confusion_matrix__{s}.png',dpi=170);plt.close(fig)
    o=test[['job_link','job_title','target_role']].copy();o['predicted_role']=p;o['correct']=o.target_role==o.predicted_role;o.to_csv(OUT/f'predictions__{s}.csv',index=False)

def tune(Xtr,ytr,Xv,yv):
    rows=[];best=(-1,None)
    for c in [.25,.5,1,2]:
        m=LinearSVC(C=c,class_weight='balanced',random_state=42);m.fit(Xtr,ytr);f=f1_score(yv,m.predict(Xv),average='macro',zero_division=0);rows.append({'C':c,'val_macro_f1':f});
        if f>best[0]:best=(f,c)
    return best[1],pd.DataFrame(rows)

print('load',flush=True);df=load_data();df,tr,va,te=split_data(df);train=df.iloc[tr];val=df.iloc[va];test=df.iloc[te];ytr=train.target_role;yv=val.target_role;yt=test.target_role
results=[];tunes=[];mats={};vecs={}
for col in ['text_unmasked','text_masked']:
    print('vectorize',col,flush=True);v=TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=2,sublinear_tf=True,strip_accents='unicode');mats[col]=(v.fit_transform(train[col]),v.transform(val[col]),v.transform(test[col]));vecs[col]=v
for col,name,note in [('text_unmasked','TF-IDF LinearSVC unmasked','Leakage diagnostic only'),('text_masked','TF-IDF LinearSVC masked','Primary leakage-controlled text model')]:
    print('fit',name,flush=True);Xtr,Xv,Xte=mats[col];c,t=tune(Xtr,ytr,Xv,yv);t.insert(0,'model',name);tunes.append(t);m=LinearSVC(C=c,class_weight='balanced',random_state=42);st=time.perf_counter();m.fit(Xtr,ytr);trs=time.perf_counter()-st;st=time.perf_counter();p=m.predict(Xte);inf=time.perf_counter()-st;results.append(metric(name,yt,p,trs,inf,note));save_diag(name,yt,p,test);joblib.dump(vs:=vecs[col],MOD/f'{slug(name)}_vectorizer.joblib');joblib.dump(m,MOD/f'{slug(name)}.joblib')
print('hybrid',flush=True);cols=['strict__'+s for s in SKILLS];Xtr_t,Xv_t,Xte_t=mats['text_masked'];Xtr=sp.hstack([Xtr_t,sp.csr_matrix(train[cols].values)],format='csr');Xv=sp.hstack([Xv_t,sp.csr_matrix(val[cols].values)],format='csr');Xte=sp.hstack([Xte_t,sp.csr_matrix(test[cols].values)],format='csr');c,t=tune(Xtr,ytr,Xv,yv);name='Hybrid masked TF-IDF + strict skills LinearSVC';t.insert(0,'model',name);tunes.append(t);m=LinearSVC(C=c,class_weight='balanced',random_state=42);st=time.perf_counter();m.fit(Xtr,ytr);trs=time.perf_counter()-st;st=time.perf_counter();p=m.predict(Xte);inf=time.perf_counter()-st;results.append(metric(name,yt,p,trs,inf,'Proposed hybrid classical model'));save_diag(name,yt,p,test);joblib.dump(vecs['text_masked'],MOD/'hybrid_vectorizer.joblib');joblib.dump(m,MOD/'hybrid_linearsvc.joblib')
pd.DataFrame(results).to_csv(OUT/'text_hybrid_results.csv',index=False);pd.concat(tunes,ignore_index=True).to_csv(OUT/'text_hybrid_validation_tuning.csv',index=False)
print(pd.DataFrame(results)[['model','accuracy','macro_f1']].to_string(index=False),flush=True)
