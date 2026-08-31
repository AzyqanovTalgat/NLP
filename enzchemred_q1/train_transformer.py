from __future__ import annotations

import argparse, gzip, json, math, os, random, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)
MODEL_NAME=os.environ.get('MODEL_NAME','microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
DATA=Path(os.environ.get('DATA_DIR','enzchemred_q1/prepared'))
OUT=Path(os.environ.get('OUTPUT_DIR','enzchemred_q1/results/transformer')); OUT.mkdir(parents=True,exist_ok=True)
DEVICE=torch.device('cpu')
LABELS=['O','B-Chemical','I-Chemical','B-Protein','I-Protein']
L2I={x:i for i,x in enumerate(LABELS)}; I2L={i:x for x,i in L2I.items()}
ENTITY_TYPES=['Chemical','Protein']

def load_jsonl(path):
    with open(path,encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]

def repair(labels):
    out=[]; prev='O'
    for t in labels:
        if t.startswith('I-') and prev not in {'B-'+t[2:],'I-'+t[2:]}: t='B-'+t[2:]
        out.append(t); prev=t
    return out

def spans(row, labels):
    labels=repair(labels); offsets=row['token_offsets']; s=set(); st=en=typ=None
    for i,t in enumerate(list(labels)+['O']):
        if t.startswith('B-'):
            if st is not None: s.add((st,en,typ))
            st,en,typ=offsets[i][0],offsets[i][1],t[2:]
        elif t.startswith('I-'):
            nt=t[2:]
            if st is None or nt!=typ:
                if st is not None: s.add((st,en,typ))
                st,en,typ=offsets[i][0],offsets[i][1],nt
            else: en=offsets[i][1]
        else:
            if st is not None: s.add((st,en,typ))
            st=en=typ=None
    return s

def ner_metrics(rows,preds):
    tp=fp=fn=0; by={t:[0,0,0] for t in ENTITY_TYPES}; yt=[]; yp=[]
    for r,pred in zip(rows,preds):
        a=spans(r,r['labels']); b=spans(r,pred); tp+=len(a&b); fp+=len(b-a); fn+=len(a-b)
        for t in ENTITY_TYPES:
            aa={x for x in a if x[2]==t}; bb={x for x in b if x[2]==t}; by[t][0]+=len(aa&bb); by[t][1]+=len(bb-aa); by[t][2]+=len(aa-bb)
        yt+=repair(r['labels']); yp+=repair(pred)
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0; F=2*P*R/(P+R) if P+R else 0
    tf={}
    for t,(a,b,c) in by.items():
        p=a/(a+b) if a+b else 0; r=a/(a+c) if a+c else 0; tf[t]=2*p*r/(p+r) if p+r else 0
    return {'precision':P,'recall':R,'exact_f1':F,'macro_type_f1':float(np.mean(list(tf.values()))),'chemical_f1':tf['Chemical'],'protein_f1':tf['Protein'],'token_accuracy':accuracy_score(yt,yp),'token_macro_f1':f1_score(yt,yp,labels=LABELS,average='macro',zero_division=0),'tp':tp,'fp':fp,'fn':fn}

def bootstrap_ner(rows,preds,n=300):
    by=defaultdict(list)
    for i,r in enumerate(rows): by[r['pmid']].append(i)
    docs=list(by); rng=np.random.default_rng(2026); vals=[]
    for _ in range(n):
        rr=[]; pp=[]
        for d in rng.choice(docs,len(docs),replace=True):
            for i in by[d]: rr.append(rows[i]); pp.append(preds[i])
        vals.append(ner_metrics(rr,pp)['exact_f1'])
    return float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))

def re_metrics(y,pred,score):
    P,R,F,_=precision_recall_fscore_support(y,pred,pos_label=1,average='binary',zero_division=0)
    return {'precision':P,'recall':R,'positive_f1':F,'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'mcc':matthews_corrcoef(y,pred),'roc_auc':roc_auc_score(y,score),'pr_auc':average_precision_score(y,score)}

def bootstrap_re(rows,y,pred,n=500):
    by=defaultdict(list)
    for i,r in enumerate(rows): by[r['pmid']].append(i)
    docs=list(by); rng=np.random.default_rng(2026); y=np.asarray(y); pred=np.asarray(pred); vals=[]
    for _ in range(n):
        idx=np.concatenate([np.asarray(by[d]) for d in rng.choice(docs,len(docs),replace=True)])
        P,R,F,_=precision_recall_fscore_support(y[idx],pred[idx],pos_label=1,average='binary',zero_division=0)
        vals.append(F)
    return float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))

def choose_threshold(y,score):
    best=(.5,-1)
    for t in np.linspace(.05,.95,181):
        f=f1_score(y,np.asarray(score)>=t,zero_division=0)
        if f>best[1]: best=(float(t),float(f))
    return best[0]

def freeze_lower_bert(model,n=6):
    base=getattr(model,'bert',None)
    if base is None: return
    for p in base.embeddings.parameters(): p.requires_grad=False
    for layer in base.encoder.layer[:n]:
        for p in layer.parameters(): p.requires_grad=False

class NERDataset(Dataset):
    def __init__(self,rows,tokenizer,max_length=256):
        self.rows=rows; self.items=[]
        for idx,r in enumerate(rows):
            enc=tokenizer(r['tokens'],is_split_into_words=True,truncation=True,max_length=max_length,add_special_tokens=True)
            word_ids=enc.word_ids(); labels=[]; first=[]; prev=None
            for pos,wid in enumerate(word_ids):
                if wid is None: labels.append(-100)
                elif wid!=prev:
                    labels.append(L2I[r['labels'][wid]]); first.append((wid,pos))
                else: labels.append(-100)
                prev=wid
            self.items.append({'input_ids':enc['input_ids'],'attention_mask':enc['attention_mask'],'labels':labels,'first':first,'idx':idx})
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def ner_collate(batch,tokenizer):
    basic=[{'input_ids':x['input_ids'],'attention_mask':x['attention_mask']} for x in batch]
    padded=tokenizer.pad(basic,padding=True,return_tensors='pt')
    max_len=int(padded['input_ids'].shape[1])
    label_tensor=torch.full((len(batch),max_len),-100,dtype=torch.long)
    for i,item in enumerate(batch): label_tensor[i,:len(item['labels'])]=torch.tensor(item['labels'],dtype=torch.long)
    padded['labels']=label_tensor
    return padded,[x['first'] for x in batch],[x['idx'] for x in batch]

def ner_predict(model,loader,rows):
    model.eval(); result=[None]*len(rows)
    with torch.no_grad():
        for batch,firsts,idxs in loader:
            batch={k:v.to(DEVICE) for k,v in batch.items()}; logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits.cpu()
            for b,(first,idx) in enumerate(zip(firsts,idxs)):
                pred=['O']*len(rows[idx]['tokens'])
                for wid,pos in first: pred[wid]=I2L[int(logits[b,pos].argmax())]
                result[idx]=pred
    return result

def train_ner():
    rows={s:load_jsonl(DATA/f'ner_{s}.jsonl') for s in ['train','dev','id_test','temporal_ood_test']}
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=True)
    ds={s:NERDataset(rows[s],tokenizer) for s in rows}
    loaders={s:DataLoader(ds[s],batch_size=12 if s=='train' else 32,shuffle=(s=='train'),collate_fn=lambda b,t=tokenizer:ner_collate(b,t),num_workers=0) for s in ds}
    model=AutoModelForTokenClassification.from_pretrained(MODEL_NAME,num_labels=len(LABELS),id2label=I2L,label2id=L2I,ignore_mismatched_sizes=True).to(DEVICE); freeze_lower_bert(model,6)
    counts=Counter(x for r in rows['train'] for x in r['labels']); total=sum(counts.values()); weights=torch.tensor([min(4.0,math.sqrt(total/(len(LABELS)*counts[l]))) for l in LABELS],dtype=torch.float)
    lossfn=nn.CrossEntropyLoss(weight=weights,ignore_index=-100,label_smoothing=.01)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5,weight_decay=.01)
    best=-1; best_state=None; history=[]; t0=time.time()
    for epoch in range(1,4):
        model.train(); losses=[]
        for step,(batch,first,idx) in enumerate(loaders['train'],1):
            opt.zero_grad(set_to_none=True); batch={k:v.to(DEVICE) for k,v in batch.items()}; logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits; loss=lossfn(logits.reshape(-1,len(LABELS)),batch['labels'].reshape(-1)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss))
            if step%150==0: print(f'NER epoch={epoch} step={step}/{len(loaders["train"])} loss={np.mean(losses[-100:]):.4f}',flush=True)
        dp=ner_predict(model,loaders['dev'],rows['dev']); met=ner_metrics(rows['dev'],dp); history.append({'epoch':epoch,'loss':float(np.mean(losses)),'dev':met}); print('NER DEV',history[-1],flush=True)
        if met['exact_f1']>best:
            best=met['exact_f1']; best_state={k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); train_seconds=time.time()-t0
    out=[]; preds={}
    for s in ['dev','id_test','temporal_ood_test']:
        p=ner_predict(model,loaders[s],rows[s]); m=ner_metrics(rows[s],p); ci=bootstrap_ner(rows[s],p) if s!='dev' else (np.nan,np.nan); out.append({'task':'NER','model':'BiomedBERT/PubMedBERT token classifier','family':'domain Transformer','split':s,**m,'ci95_low':ci[0],'ci95_high':ci[1],'train_seconds':train_seconds,'best_dev_f1':best,'epochs':3,'hyperparameters':json.dumps({'model':MODEL_NAME,'max_length':256,'batch':12,'lr':3e-5,'frozen_lower_layers':6})}); preds[s]=p
    pd.DataFrame(out).to_csv(OUT/'ner_transformer_results.csv',index=False)
    with gzip.open(OUT/'ner_transformer_predictions.json.gz','wt',encoding='utf-8') as f: json.dump({'BiomedBERT/PubMedBERT token classifier':preds},f)
    (OUT/'ner_transformer_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return out

class REDataset(Dataset):
    def __init__(self,rows,tokenizer,max_length=256):
        self.rows=rows; self.enc=tokenizer([r['marked_sentence'] for r in rows],truncation=True,max_length=max_length,padding=False); self.labels=[int(r['binary_label']) for r in rows]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return {k:v[i] for k,v in self.enc.items()}|{'labels':self.labels[i],'idx':i}

def re_collate(batch,tokenizer):
    idx=[x.pop('idx') for x in batch]; labels=torch.tensor([x.pop('labels') for x in batch],dtype=torch.long); padded=tokenizer.pad(batch,padding=True,return_tensors='pt'); padded['labels']=labels; return padded,idx

def re_predict(model,loader,n):
    model.eval(); scores=np.zeros(n,dtype=np.float32)
    with torch.no_grad():
        for batch,idx in loader:
            batch.pop('labels'); batch={k:v.to(DEVICE) for k,v in batch.items()}; prob=torch.softmax(model(**batch).logits,dim=-1)[:,1].cpu().numpy(); scores[idx]=prob
    return scores

def train_re():
    rows={s:load_jsonl(DATA/f're_{s}.jsonl') for s in ['train','dev','id_test','temporal_ood_test']}
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=True); special=['<C1>','</C1>','<C2>','</C2>']; tokenizer.add_special_tokens({'additional_special_tokens':special})
    ds={s:REDataset(rows[s],tokenizer) for s in rows}; loaders={s:DataLoader(ds[s],batch_size=16 if s=='train' else 48,shuffle=(s=='train'),collate_fn=lambda b,t=tokenizer:re_collate(b,t),num_workers=0) for s in ds}
    model=AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=2,ignore_mismatched_sizes=True).to(DEVICE); model.resize_token_embeddings(len(tokenizer)); freeze_lower_bert(model,6)
    pos=sum(r['binary_label'] for r in rows['train']); neg=len(rows['train'])-pos; weights=torch.tensor([1.0,math.sqrt(neg/pos)],dtype=torch.float); lossfn=nn.CrossEntropyLoss(weight=weights,label_smoothing=.01); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2.5e-5,weight_decay=.01)
    best=-1; best_state=None; best_thr=.5; history=[]; t0=time.time()
    for epoch in range(1,4):
        model.train(); losses=[]
        for step,(batch,idx) in enumerate(loaders['train'],1):
            labels=batch.pop('labels').to(DEVICE); batch={k:v.to(DEVICE) for k,v in batch.items()}; opt.zero_grad(set_to_none=True); logits=model(**batch).logits; loss=lossfn(logits,labels); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss))
            if step%150==0: print(f'RE epoch={epoch} step={step}/{len(loaders["train"])} loss={np.mean(losses[-100:]):.4f}',flush=True)
        sc=re_predict(model,loaders['dev'],len(rows['dev'])); yy=np.array([r['binary_label'] for r in rows['dev']]); th=choose_threshold(yy,sc); met=re_metrics(yy,(sc>=th).astype(int),sc); history.append({'epoch':epoch,'loss':float(np.mean(losses)),'threshold':th,'dev':met}); print('RE DEV',history[-1],flush=True)
        if met['positive_f1']>best:
            best=met['positive_f1']; best_thr=th; best_state={k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); train_seconds=time.time()-t0
    out=[]; preds={}
    for s in ['dev','id_test','temporal_ood_test']:
        sc=re_predict(model,loaders[s],len(rows[s])); yy=np.array([r['binary_label'] for r in rows[s]]); pp=(sc>=best_thr).astype(int); met=re_metrics(yy,pp,sc); ci=bootstrap_re(rows[s],yy,pp) if s!='dev' else (np.nan,np.nan); out.append({'task':'RE','model':'BiomedBERT/PubMedBERT entity-marker classifier','family':'domain Transformer','split':s,**met,'ci95_low':ci[0],'ci95_high':ci[1],'threshold':best_thr,'train_seconds':train_seconds,'best_dev_f1':best,'epochs':3,'hyperparameters':json.dumps({'model':MODEL_NAME,'max_length':256,'batch':16,'lr':2.5e-5,'frozen_lower_layers':6,'markers':special})}); preds[s]={'pred':pp.tolist(),'score':sc.tolist()}
    pd.DataFrame(out).to_csv(OUT/'re_transformer_results.csv',index=False)
    with gzip.open(OUT/'re_transformer_predictions.json.gz','wt',encoding='utf-8') as f: json.dump({'BiomedBERT/PubMedBERT entity-marker classifier':preds},f)
    (OUT/'re_transformer_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('task',choices=['ner','re']); args=ap.parse_args(); result=train_ner() if args.task=='ner' else train_re(); (OUT/f'{args.task}_complete.json').write_text(json.dumps({'status':'success','task':args.task,'model':MODEL_NAME,'results':result},indent=2),encoding='utf-8')

if __name__=='__main__': main()
