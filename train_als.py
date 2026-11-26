"""Matrix-factorization recommender (implicit ALS), trained for 5 rounds.
Inputs: events_train.csv / events_holdout.csv (viewing history) + catalog.csv.
Outputs in model/: als_model.npz (member + title vectors), meta.json, report in REPORT.md.
Hit Rate@10: is the member's newest hidden title inside their top-10 picks?"""
import csv,os,json,itertools,numpy as np,wandb
for l in open('.env'):
    if l.startswith('WANDB_API_KEY='):os.environ['WANDB_API_KEY']=l.split('=',1)[1].strip()
K,ROUNDS,VAL_START,SEED=10,5,"2026-11-03",0
rd=lambda p:list(csv.DictReader(open(p,encoding='utf-8')))
tr=rd('events_train.csv');ho=rd('events_holdout.csv');cat=rd('catalog.csv')
tid=[r['title_id'] for r in cat];ix={t:i for i,t in enumerate(tid)};N=len(tid)
def targets(ev):
    t={}
    for r in sorted(ev,key=lambda r:(r['event_datetime'],r['event_id'])):t[r['member_id']]=ix[r['title_id']]
    return t
def matrix(ev):
    ms=sorted({r['member_id'] for r in ev});mi={m:i for i,m in enumerate(ms)}
    M=np.zeros((len(ms),N))
    for r in ev:M[mi[r['member_id']],ix[r['title_id']]]+=1
    return ms,mi,M
def als(M,f,reg,alpha,rounds,log=None):
    rng=np.random.default_rng(SEED);U=rng.normal(0,.1,(M.shape[0],f));V=rng.normal(0,.1,(N,f))
    P=(M>0).astype(float);C=1+alpha*np.log1p(M)
    def solve(X,Pm,Cm):
        out=np.zeros((Pm.shape[0],f));XtX=X.T@X
        for i in range(Pm.shape[0]):
            c=Cm[i];A=XtX+(X.T*(c-1))@X+reg*np.eye(f);out[i]=np.linalg.solve(A,(X.T*c)@Pm[i])
        return out
    for r in range(1,rounds+1):
        U=solve(V,P,C);V=solve(U,P.T,C.T)
        if log:log(r,U,V)
    return U,V
def hit(S,mi,tg):
    return float(np.mean([t in np.argsort(-S[mi[m]])[:K] for m,t in tg.items() if m in mi]))
def pop_scores(M,n):
    p=(M>0).sum(0);return np.tile(p,(n,1)).astype(float)   # same top-10 for everyone
fit=[r for r in tr if r['event_datetime']<VAL_START];val=targets([r for r in tr if r['event_datetime']>=VAL_START])
ms,mi,M=matrix(fit)
run=wandb.init(project="netflix-next-title",name="als-5-rounds",config=dict(K=K,rounds=ROUNDS,val_start=VAL_START))
best=None
for f,reg,alpha in itertools.product([8,16,32],[0.1,1,10],[5,20,40]):
    U,V=als(M,f,reg,alpha,ROUNDS);h=hit(U@V.T,mi,val)
    if best is None or h>best[0]:best=(h,f,reg,alpha)
vh,f,reg,alpha=best;print('chosen',dict(factors=f,reg=reg,alpha=alpha),'val hit@10',round(vh,3))
popv=hit(pop_scores(M,len(ms)),mi,val)
run.config.update(dict(factors=f,reg=reg,alpha=alpha))
# final: train on ALL training history (before the cutoff), log each of the 5 rounds
ms,mi,M=matrix(tr);tg=targets(ho);rounds=[]
def log(r,U,V):
    S=U@V.T;row=dict(round=r,holdout_hit10_model=hit(S,mi,tg),holdout_hit10_popular=hit(pop_scores(M,len(ms)),mi,tg))
    rounds.append(row);wandb.log(row);print(row)
U,V=als(M,f,reg,alpha,ROUNDS,log)
S=U@V.T;mh,ph=hit(S,mi,tg),hit(pop_scores(M,len(ms)),mi,tg)
os.makedirs('model',exist_ok=True)
np.savez_compressed('model/als_model.npz',member_vectors=U,title_vectors=V,member_ids=np.array(ms),title_ids=np.array(tid),
    seen=(M>0))
json.dump(dict(model='implicit ALS',rounds=ROUNDS,factors=f,reg=reg,alpha=alpha,k=K,train_cutoff='2026-11-10',
    n_members=len(ms),n_titles=N,n_test_members=len(tg),hit10_model=mh,hit10_popular=ph,val_hit10=vh,val_hit10_popular=popv,
    per_round=rounds),open('model/meta.json','w'),indent=1)
# top-10 picks per member
top={m:[tid[j] for j in np.argsort(-S[mi[m]])[:K]] for m in ms}
json.dump(top,open('model/top10_picks.json','w'))
n=len(tg);run.summary.update(dict(hit10_model=mh,hit10_popular=ph));run.finish()
open('REPORT.md','w').write(f"""# Hit Rate@10 on the hidden week (10-16 Nov 2026)
{n} members; a hit = the member's newest hidden title is in their top 10.

| | Hit Rate@10 |
|---|---|
| Our model (ALS, {ROUNDS} rounds, {f} factors) | **{mh:.1%}** ({round(mh*n)}/{n}) |
| Popular top 10 (same list for everyone) | {ph:.1%} ({round(ph*n)}/{n}) |

Per training round (holdout): """+", ".join(f"r{x['round']}: {x['holdout_hit10_model']:.1%}" for x in rounds)+f"""

Hyperparameters (factors={f}, reg={reg}, alpha={alpha}) were chosen on a validation week (3-9 Nov) cut from the end of the training data
(val Hit@10 {vh:.1%} vs popular {popv:.1%}); the holdout week was not used for tuning.
Load with `np.load('model/als_model.npz')`: member_vectors, title_vectors, member_ids, title_ids. Score = member_vector @ title_vectors.T.
""")
print('HOLDOUT model',round(mh,3),'popular',round(ph,3))
