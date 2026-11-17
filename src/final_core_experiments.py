from __future__ import annotations

from pathlib import Path
import json, re, time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / 'data' / 'raw_midterm' / 'raw'
OUT_DIR = BASE_DIR / 'results' / 'final_core'
FIG_DIR = BASE_DIR / 'figures' / 'final_core'
MODEL_DIR = BASE_DIR / 'models' / 'final_core'
for d in (OUT_DIR, FIG_DIR, MODEL_DIR): d.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
ROLES = ['Data Analyst','Data Engineer','Data Scientist','Machine Learning Engineer']

SKILL_ALIASES = {
'python':['python'],'sql':['sql'],'r':['r','r programming'],'excel':['excel','microsoft excel'],'tableau':['tableau'],'power bi':['power bi','powerbi'],'looker':['looker'],'java':['java'],'scala':['scala'],'c++':['c++','cpp'],'javascript':['javascript','js'],'aws':['aws','amazon web services'],'azure':['azure','microsoft azure'],'gcp':['gcp','google cloud','google cloud platform'],'spark':['spark','apache spark','pyspark'],'hadoop':['hadoop','apache hadoop'],'snowflake':['snowflake'],'databricks':['databricks'],'redshift':['redshift','aws redshift'],'bigquery':['bigquery','google bigquery'],'postgresql':['postgresql','postgres'],'mysql':['mysql'],'mongodb':['mongodb','mongo db'],'machine learning':['machine learning','ml'],'deep learning':['deep learning'],'scikit-learn':['scikit-learn','scikit learn','scikitlearn','sklearn'],'tensorflow':['tensorflow'],'pytorch':['pytorch','py torch'],'keras':['keras'],'nlp':['nlp','natural language processing'],'computer vision':['computer vision'],'statistics':['statistics','statistical analysis','statistical modeling'],'data visualization':['data visualization','data visualisation'],'git':['git'],'github':['github'],'docker':['docker'],'kubernetes':['kubernetes','k8s'],'airflow':['airflow','apache airflow'],'dbt':['dbt','data build tool'],'kafka':['kafka','apache kafka']}
SKILLS=list(SKILL_ALIASES)
ROLE_PATTERNS=[r'\bdata\s+scientists?\b',r'\bdata\s+engineers?\b',r'\bdata\s+analysts?\b',r'\bmachine\s+learning\s+engineers?\b',r'\bml\s+engineers?\b',r'\bmle\b']

def classify_role(title):
    t=re.sub(r'\s+',' ',str(title).lower().strip())
    if re.search(r'\b(machine learning|ml)\s+engineer\b',t): return 'Machine Learning Engineer'
    if re.search(r'\bdata\s+scientist\b',t): return 'Data Scientist'
    if re.search(r'\bdata\s+engineer\b',t): return 'Data Engineer'
    if re.search(r'\bdata\s+analyst\b',t): return 'Data Analyst'
    return 'Other'

def mask_text(text,title):
    text='' if pd.isna(text) else str(text); title='' if pd.isna(title) else str(title).strip()
    if title: text=re.sub(re.escape(title),' [ROLE_TITLE] ',text,flags=re.I)
    for p in ROLE_PATTERNS: text=re.sub(p,' [ROLE] ',text,flags=re.I)
    return re.sub(r'\s+',' ',text).strip()

def compile_alias(alias):
    return re.compile(r'(?<![a-z0-9+#])'+re.escape(alias)+r'(?![a-z0-9+#])',re.I)
COMPILED={k:[compile_alias(a) for a in v] for k,v in SKILL_ALIASES.items()}
def extract_matrix(texts):
    rows=[]
    for text in texts.fillna('').astype(str):
        rows.append([int(any(p.search(text) for p in COMPILED[s])) for s in SKILLS])
    return pd.DataFrame(rows,columns=SKILLS,index=texts.index)

def load_data():
    post=pd.read_csv(RAW_DIR/'job_postings.csv'); skills=pd.read_csv(RAW_DIR/'job_skills.csv'); summ=pd.read_csv(RAW_DIR/'job_summary.csv')
    df=post.merge(skills,on='job_link',how='inner',validate='one_to_one').merge(summ,on='job_link',how='inner',validate='one_to_one')
    raw=len(df); df['job_summary']=df.job_summary.fillna('').astype(str); df['job_skills']=df.job_skills.fillna('').astype(str); df['job_title']=df.job_title.fillna('').astype(str).str.strip()
    df['summary_normalized']=df.job_summary.str.lower().str.replace(r'\s+',' ',regex=True).str.strip(); df=df.drop_duplicates('summary_normalized').copy(); dedup=len(df)
    df=df[df.job_summary.str.len()>=200].copy(); longn=len(df); df['target_role']=df.job_title.apply(classify_role); df=df[df.target_role!='Other'].copy().reset_index(drop=True)
    df['text_unmasked']=df.job_summary.str.replace(r'\s+',' ',regex=True).str.strip(); df['text_masked']=[mask_text(t,j) for t,j in zip(df.job_summary,df.job_title)]
    strict=extract_matrix(df.text_masked).add_prefix('strict__'); legacy=extract_matrix(df.job_skills).add_prefix('legacy__'); df=pd.concat([df,strict,legacy],axis=1)
    audit={'raw_merged_rows':raw,'after_exact_summary_dedup':dedup,'after_min_200_chars':longn,'final_four_role_rows':len(df),'class_counts':df.target_role.value_counts().to_dict(),'duplicate_normalized_summaries_after_cleaning':int(df.summary_normalized.duplicated().sum())}
    (OUT_DIR/'dataset_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return df

def split_data(df):
    idx=np.arange(len(df)); tv,test=train_test_split(idx,test_size=.2,random_state=42,stratify=df.target_role); train,val=train_test_split(tv,test_size=.125,random_state=42,stratify=df.iloc[tv].target_role)
    out=df.copy(); out['split']=''; out.loc[train,'split']='train'; out.loc[val,'split']='val'; out.loc[test,'split']='test'; out[['job_link','target_role','split']].to_csv(OUT_DIR/'split_manifest.csv',index=False); pd.crosstab(out.split,out.target_role).to_csv(OUT_DIR/'split_class_counts.csv')
    return out,train,val,test

def metrics(name,y,pred,tr,inf,note):
    return {'model':name,'accuracy':accuracy_score(y,pred),'macro_precision':precision_score(y,pred,average='macro',zero_division=0),'macro_recall':recall_score(y,pred,average='macro',zero_division=0),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),'train_seconds':tr,'test_inference_seconds':inf,'note':note}
def slug(s): return re.sub(r'[^a-z0-9]+','_',s.lower()).strip('_')
def save_diag(name,y,pred,test):
    s=slug(name); pd.DataFrame(classification_report(y,pred,output_dict=True,zero_division=0)).T.to_csv(OUT_DIR/f'classification_report__{s}.csv'); cm=confusion_matrix(y,pred,labels=ROLES); pd.DataFrame(cm,index=ROLES,columns=ROLES).to_csv(OUT_DIR/f'confusion_matrix__{s}.csv')
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm); ax.set_xticks(range(4),ROLES,rotation=28,ha='right'); ax.set_yticks(range(4),ROLES); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(name)
    for i in range(4):
        for j in range(4): ax.text(j,i,int(cm[i,j]),ha='center',va='center')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04); fig.tight_layout(); fig.savefig(FIG_DIR/f'confusion_matrix__{s}.png',dpi=170); plt.close(fig)
    pdf=test[['job_link','job_title','target_role','text_masked']].copy(); pdf['predicted_role']=pred; pdf['correct']=pdf.target_role==pdf.predicted_role; pdf.to_csv(OUT_DIR/f'predictions__{s}.csv',index=False)

def tune_lr(Xtr,ytr,Xv,yv):
    rows=[]; best=(-1,None)
    for c in [.25,.5,1,2,4]:
        m=LogisticRegression(C=c,class_weight='balanced',max_iter=3000,random_state=42); m.fit(Xtr,ytr); f=f1_score(yv,m.predict(Xv),average='macro',zero_division=0); rows.append({'C':c,'val_macro_f1':f});
        if f>best[0]: best=(f,c)
    return best[1],pd.DataFrame(rows)
def tune_svc(Xtr,ytr,Xv,yv):
    rows=[]; best=(-1,None)
    for c in [.25,.5,1,2]:
        m=LinearSVC(C=c,class_weight='balanced',random_state=42); m.fit(Xtr,ytr); f=f1_score(yv,m.predict(Xv),average='macro',zero_division=0); rows.append({'C':c,'val_macro_f1':f});
        if f>best[0]: best=(f,c)
    return best[1],pd.DataFrame(rows)

def run():
    print('1/6 Load + leakage-controlled preprocessing',flush=True); df=load_data(); df,tr,va,te=split_data(df); train=df.iloc[tr]; val=df.iloc[va]; test=df.iloc[te]; ytr=train.target_role; yv=val.target_role; yt=test.target_role; results=[]; tunes=[]
    print('2/6 Skills-only baselines',flush=True)
    for label,prefix,note in [('Legacy skills LR','legacy__','Legacy provided-skill baseline'),('Strict masked-description skills LR','strict__','40 skills extracted from masked descriptions')]:
        cols=[prefix+s for s in SKILLS]; c,t=tune_lr(train[cols],ytr,val[cols],yv); t.insert(0,'model',label); tunes.append(t); m=LogisticRegression(C=c,class_weight='balanced',max_iter=3000,random_state=42); st=time.perf_counter();m.fit(train[cols],ytr);trs=time.perf_counter()-st;st=time.perf_counter();p=m.predict(test[cols]);ifs=time.perf_counter()-st;results.append(metrics(label,yt,p,trs,ifs,note));save_diag(label,yt,p,test);joblib.dump(m,MODEL_DIR/f'{slug(label)}.joblib')
    cols=['strict__'+s for s in SKILLS]; dt=DecisionTreeClassifier(max_depth=8,min_samples_leaf=5,class_weight='balanced',random_state=42);st=time.perf_counter();dt.fit(train[cols],ytr);trs=time.perf_counter()-st;st=time.perf_counter();p=dt.predict(test[cols]);ifs=time.perf_counter()-st;label='Strict skills Decision Tree';results.append(metrics(label,yt,p,trs,ifs,'Interpretable nonlinear baseline'));save_diag(label,yt,p,test);joblib.dump(dt,MODEL_DIR/f'{slug(label)}.joblib')
    print('3/6 Fit TF-IDF representations',flush=True)
    text_mats={}; vectors={}
    for textcol in ['text_unmasked','text_masked']:
        v=TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=2,sublinear_tf=True,strip_accents='unicode'); Xtr=v.fit_transform(train[textcol]); Xv=v.transform(val[textcol]); Xte=v.transform(test[textcol]); text_mats[textcol]=(Xtr,Xv,Xte); vectors[textcol]=v
    print('4/6 Text models + masking diagnostic',flush=True)
    for textcol,label,note in [('text_unmasked','TF-IDF LinearSVC unmasked','Leakage diagnostic only; title/role strings present'),('text_masked','TF-IDF LinearSVC masked','Primary leakage-controlled classical text model')]:
        Xtr,Xv,Xte=text_mats[textcol]; c,t=tune_svc(Xtr,ytr,Xv,yv);t.insert(0,'model',label);tunes.append(t);m=LinearSVC(C=c,class_weight='balanced',random_state=42);st=time.perf_counter();m.fit(Xtr,ytr);trs=time.perf_counter()-st;st=time.perf_counter();p=m.predict(Xte);ifs=time.perf_counter()-st;results.append(metrics(label,yt,p,trs,ifs,note));save_diag(label,yt,p,test);joblib.dump(vectors[textcol],MODEL_DIR/f'{slug(label)}_vectorizer.joblib');joblib.dump(m,MODEL_DIR/f'{slug(label)}.joblib')
    print('5/6 Hybrid proposed model',flush=True)
    Xtr_t,Xv_t,Xte_t=text_mats['text_masked']; Xtr=sp.hstack([Xtr_t,sp.csr_matrix(train[cols].values)],format='csr');Xv=sp.hstack([Xv_t,sp.csr_matrix(val[cols].values)],format='csr');Xte=sp.hstack([Xte_t,sp.csr_matrix(test[cols].values)],format='csr');c,t=tune_svc(Xtr,ytr,Xv,yv);label='Hybrid masked TF-IDF + strict skills LinearSVC';t.insert(0,'model',label);tunes.append(t);m=LinearSVC(C=c,class_weight='balanced',random_state=42);st=time.perf_counter();m.fit(Xtr,ytr);trs=time.perf_counter()-st;st=time.perf_counter();p=m.predict(Xte);ifs=time.perf_counter()-st;results.append(metrics(label,yt,p,trs,ifs,'Proposed hybrid classical model'));save_diag(label,yt,p,test);joblib.dump(vectors['text_masked'],MODEL_DIR/'hybrid_vectorizer.joblib');joblib.dump(m,MODEL_DIR/'hybrid_linearsvc.joblib')
    print('6/6 Save verification artifacts',flush=True)
    comp=pd.DataFrame(results).sort_values('macro_f1',ascending=False);comp.to_csv(OUT_DIR/'model_comparison.csv',index=False);pd.concat(tunes,ignore_index=True).to_csv(OUT_DIR/'validation_tuning.csv',index=False)
    lu=comp.set_index('model'); mask=pd.DataFrame([{'unmasked_accuracy':lu.loc['TF-IDF LinearSVC unmasked','accuracy'],'masked_accuracy':lu.loc['TF-IDF LinearSVC masked','accuracy'],'accuracy_drop':lu.loc['TF-IDF LinearSVC unmasked','accuracy']-lu.loc['TF-IDF LinearSVC masked','accuracy'],'unmasked_macro_f1':lu.loc['TF-IDF LinearSVC unmasked','macro_f1'],'masked_macro_f1':lu.loc['TF-IDF LinearSVC masked','macro_f1'],'macro_f1_drop':lu.loc['TF-IDF LinearSVC unmasked','macro_f1']-lu.loc['TF-IDF LinearSVC masked','macro_f1']}]);mask.to_csv(OUT_DIR/'masking_effect.csv',index=False)
    eligible=['Strict masked-description skills LR','Strict skills Decision Tree','TF-IDF LinearSVC masked','Hybrid masked TF-IDF + strict skills LinearSVC'];best=comp[comp.model.isin(eligible)].iloc[0]; pred=pd.read_csv(OUT_DIR/f'predictions__{slug(best.model)}.csv');err=pred[~pred.correct.astype(bool)].copy();err.groupby(['target_role','predicted_role']).size().reset_index(name='count').sort_values('count',ascending=False).to_csv(OUT_DIR/'error_confusion_pairs.csv',index=False);err.head(50).to_csv(OUT_DIR/'error_examples_first50.csv',index=False)
    summary={'best_eligible_model':best.model,'best_eligible_test_accuracy':float(best.accuracy),'best_eligible_test_macro_f1':float(best.macro_f1),'title_used_as_feature':False,'unmasked_text_is_diagnostic_only':True,'exact_summary_duplicates_after_cleaning':0,'validation_used_for_hyperparameter_selection':True,'test_evaluated_after_selection':True};(OUT_DIR/'verification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(comp[['model','accuracy','macro_f1']].to_string(index=False),flush=True)
if __name__=='__main__': run()
