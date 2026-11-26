"""Next-title recommender. Hit@10: is each member's newest hidden title in their top 10?
Model chosen on a validation week carved from the END of train (3-9 Nov); the holdout
week (10-16 Nov) is scored once, at the end."""
import csv,os,itertools,numpy as np
import wandb
for l in open('.env'):
    if l.startswith('WANDB_API_KEY='):os.environ['WANDB_API_KEY']=l.split('=',1)[1].strip()
K=10;VAL_START="2026-11-03"
rd=lambda p:list(csv.DictReader(open(p,encoding='utf-8')))
tr=rd('events_train.csv');ho=rd('events_holdout.csv');cat=rd('catalog.csv')
tid=[r['title_id'] for r in cat];ix={t:i for i,t in enumerate(tid)};N=len(tid)
genre=sorted({r['primary_genre'] for r in cat});gi={g:i for i,g in enumerate(genre)}
G=np.zeros((N,len(genre)));[G.__setitem__((ix[r['title_id']],gi[r['primary_genre']]),1) for r in cat]
def targets(ev):
    t={}
    for r in sorted(ev,key=lambda r:(r['event_datetime'],r['event_id'])):t[r['member_id']]=ix[r['title_id']]
    return t
def build(ev):
    ms=sorted({r['member_id'] for r in ev});mi={m:i for i,m in enumerate(ms)}
    M=np.zeros((len(ms),N))
    for r in ev:M[mi[r['member_id']],ix[r['title_id']]]+=1
    return mi,M
def scores(ev,w):
    mi,M=build(ev);B=(M>0).astype(float)
    pop=B.sum(0);pop=pop/pop.max()
    nrm=np.sqrt((B**2).sum(0))+1e-9;S=(B.T@B)/np.outer(nrm,nrm);np.fill_diagonal(S,0)
    cooc=B@S;cooc=cooc/(cooc.max(1,keepdims=True)+1e-9)
    own=M/(M.max(1,keepdims=True)+1e-9)
    gp=B@G;gp=gp/(gp.sum(1,keepdims=True)+1e-9);gen=gp@G.T
    return mi,w[0]*own+w[1]*cooc+w[2]*pop+w[3]*gen
def hit(ev,tg,w):
    mi,Sc=scores(ev,w)
    return np.mean([t in np.argsort(-Sc[mi[m]])[:K] for m,t in tg.items() if m in mi])
fit=[r for r in tr if r['event_datetime']<VAL_START];val=targets([r for r in tr if r['event_datetime']>=VAL_START])
run=wandb.init(project="netflix-next-title",config=dict(K=K,val_start=VAL_START,train_rows=len(tr),holdout_rows=len(ho)))
base={'popularity':(0,0,1,0),'own_history':(1,0,0,0),'co-watch':(0,1,0,0),'genre':(0,0,0,1)}
for n,w in base.items():
    v=hit(fit,val,w);print(n,'val hit@10',round(v,3));wandb.log({f'val_hit10/{n}':v})
best=None
for w in itertools.product([0,1,2,3],[0,1,2,3],[0,1,2],[0,1,2]):
    if not any(w):continue
    v=hit(fit,val,w)
    if best is None or v>best[0]:best=(v,w)
print('best blend (own,cooc,pop,genre)',best[1],'val hit@10',round(best[0],3))
wandb.log({'val_hit10/blend':best[0]});run.config.update({'blend_weights':best[1]})
tg=targets(ho);res={}
for n,w in {**base,'blend':best[1]}.items():res[n]=hit(tr,tg,w)
rand=K/N
print('HOLDOUT hit@10 (',len(tg),'members)',{k:round(v,3) for k,v in res.items()},'random',round(rand,3))
for n,v in res.items():wandb.log({f'test_hit10/{n}':v}); run.summary[f'test_hit10_{n}']=v
run.summary['test_hit10_random']=rand;run.summary['n_test_members']=len(tg);run.finish()
