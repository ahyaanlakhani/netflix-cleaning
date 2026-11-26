"""PyTorch two-tower next-title recommender, 64-d towers, 5 training rounds (epochs).
Title tower input = catalog DESCRIPTION (TF-IDF) + type/genre/maturity one-hots, so titles with zero plays
(e.g. TT-0300) still get a vector. Member tower input = mean of the features of titles the member watched.
Hit Rate@10: is each member's newest hidden title in their top 10?"""
import csv,os,re,json,math,collections as C,numpy as np,torch,torch.nn as nn,torch.nn.functional as F,wandb
for l in open('.env'):
    if l.startswith('WANDB_API_KEY='):os.environ['WANDB_API_KEY']=l.split('=',1)[1].strip()
K,DIM,ROUNDS,VAL_START,CUTOFF,SEED=10,64,5,"2026-11-03","2026-11-10",0
torch.manual_seed(SEED);np.random.seed(SEED)
rd=lambda p:list(csv.DictReader(open(p,encoding='utf-8')))
tr=rd('events_train.csv');ho=rd('events_holdout.csv');cat=rd('catalog.csv')
tid=[r['title_id'] for r in cat];ix={t:i for i,t in enumerate(tid)};N=len(tid)
# ---- title features from catalog description (+ metadata) ----
tok=lambda s:re.findall(r"[a-z']{3,}",s.lower())
docs=[tok(r['description']+' '+r['title_name']) for r in cat]
df=C.Counter(w for d in docs for w in set(d));vocab=[w for w,c in df.items() if c>=2];vi={w:i for i,w in enumerate(vocab)}
X=np.zeros((N,len(vocab)),dtype=np.float32)
for i,d in enumerate(docs):
    for w,c in C.Counter(d).items():
        if w in vi:X[i,vi[w]]=(1+math.log(c))*math.log(N/df[w])
X/=np.linalg.norm(X,axis=1,keepdims=True)+1e-9
def onehot(vals):
    u=sorted(set(vals));m=np.zeros((N,len(u)),dtype=np.float32)
    for i,v in enumerate(vals):m[i,u.index(v)]=1
    return m
meta=np.hstack([onehot([r['title_type'] for r in cat]),onehot([r['primary_genre'] for r in cat]),
                onehot([r['secondary_genre'] or 'none' for r in cat]),onehot([r['maturity_rating'] for r in cat])])
TF=torch.tensor(np.hstack([X,meta]))               # N x features: the title tower's ONLY input
class Tower(nn.Module):
    def __init__(s,d):super().__init__();s.n=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,DIM))
    def forward(s,x):return F.normalize(s.n(x),dim=-1)
title_t,member_t=Tower(TF.shape[1]),Tower(TF.shape[1]);temp=nn.Parameter(torch.tensor(math.log(1/0.07)))
def history_vec(titles):                            # member input: mean feature vector of watched titles
    v=TF[titles].mean(0);return v
def samples(ev):                                    # (history titles so far, next title) per event
    by=C.defaultdict(list)
    for r in sorted(ev,key=lambda r:(r['event_datetime'],r['event_id'])):by[r['member_id']].append(ix[r['title_id']])
    out=[]
    for m,s in by.items():
        for k in range(1,len(s)):out.append((s[:k],s[k]))
    return out,by
def targets(ev):
    t={}
    for r in sorted(ev,key=lambda r:(r['event_datetime'],r['event_id'])):t[r['member_id']]=ix[r['title_id']]
    return t
def train(ev,log=None):
    global title_t,member_t,temp
    torch.manual_seed(SEED);title_t,member_t=Tower(TF.shape[1]),Tower(TF.shape[1]);temp=nn.Parameter(torch.tensor(math.log(1/0.07)))
    opt=torch.optim.Adam(list(title_t.parameters())+list(member_t.parameters())+[temp],lr=2e-3,weight_decay=1e-5)
    S,_=samples(ev);H=torch.stack([history_vec(h) for h,_ in S]);Y=torch.tensor([t for _,t in S])
    for r in range(1,ROUNDS+1):
        title_t.train();member_t.train();perm=torch.randperm(len(S));tot=0
        for b in range(0,len(S),256):
            j=perm[b:b+256];u=member_t(H[j]);v=title_t(TF)         # full-catalog softmax
            loss=F.cross_entropy(u@v.T*temp.exp().clamp(max=100),Y[j]);opt.zero_grad();loss.backward();opt.step();tot+=loss.item()*len(j)
        if log:log(r,tot/len(S))
def vectors(ev):
    title_t.eval();member_t.eval();_,by=samples(ev)
    with torch.no_grad():
        ms=sorted(by);U=member_t(torch.stack([history_vec(by[m]) for m in ms])).numpy();V=title_t(TF).numpy()
    return ms,U,V
def hit(ms,S,tg,tops=None):
    mi={m:i for i,m in enumerate(ms)}
    return float(np.mean([t in np.argsort(-S[mi[m]])[:K] for m,t in tg.items() if m in mi]))
def unseen(ev,ms,S):                               # titles a member played in ev can never be recommended
    mi={m:i for i,m in enumerate(ms)};S=S.astype(float).copy()
    for r in ev:
        if r['member_id'] in mi:S[mi[r['member_id']],ix[r['title_id']]]=-np.inf
    return S
def popular(ev,n):
    p=np.zeros(N)
    for r in ev:p[ix[r['title_id']]]+=1
    return np.tile(p,(n,1))
run=wandb.init(project="netflix-next-title",name="two-tower-64d-5rounds",config=dict(dim=DIM,rounds=ROUNDS,k=K,cutoff=CUTOFF,val_start=VAL_START))
fit=[r for r in tr if r['event_datetime']<VAL_START];val=targets([r for r in tr if r['event_datetime']>=VAL_START])
train(fit);ms,U,V=vectors(fit);vh,vp=hit(ms,unseen(fit,ms,U@V.T),val),hit(ms,unseen(fit,ms,popular(fit,len(ms))),val);print('val model',round(vh,3),'popular',round(vp,3))
tg=targets(ho);rounds=[]
def log(r,loss):
    ms,U,V=vectors(tr);row=dict(round=r,train_loss=loss,holdout_hit10_model=hit(ms,unseen(tr,ms,U@V.T),tg),holdout_hit10_popular=hit(ms,unseen(tr,ms,popular(tr,len(ms))),tg))
    rounds.append(row);wandb.log(row);print(row)
train(tr,log);ms,U,V=vectors(tr);S=unseen(tr,ms,U@V.T);mh,ph=hit(ms,S,tg),hit(ms,unseen(tr,ms,popular(tr,len(ms))),tg)
plays=C.Counter(r['title_id'] for r in tr);zero=[t for t in tid if plays[t]==0]
n=len(tg);top={m:[tid[j] for j in np.argsort(-S[i])[:K]] for i,m in enumerate(ms)}
in_picks=sum(t in p for p in top.values() for t in ['TT-0300'])
os.makedirs('model',exist_ok=True)
np.savez_compressed('model/twotower_vectors.npz',member_vectors=U,title_vectors=V,member_ids=np.array(ms),title_ids=np.array(tid))
torch.save(dict(title_tower=title_t.state_dict(),member_tower=member_t.state_dict(),temp=temp.data,vocab=vocab,feature_dim=TF.shape[1],dim=DIM),'model/twotower.pt')
np.save('model/title_features.npy',TF.numpy())
json.dump(top,open('model/top10_picks.json','w'))
url=run.url;json.dump(dict(model='PyTorch two-tower',dim=DIM,rounds=ROUNDS,k=K,cutoff=CUTOFF,n_scored_members=n,hit10_model=mh,hit10_popular=ph,
  val_hit10=vh,val_hit10_popular=vp,zero_play_titles=zero,tt0300_vector_norm=float(np.linalg.norm(V[ix['TT-0300']])),
  tt0300_in_n_members_top10=in_picks,per_round=rounds,wandb_run=url),open('model/meta.json','w'),indent=1)
run.summary.update(dict(hit10_model=mh,hit10_popular=ph));run.finish()
open('REPORT.md','w').write(f"""# Hit Rate@10, hidden final week
Cutoff date: **{CUTOFF}** (training = events before it; hidden week = 10-16 Nov 2026). Scored members: **{n}**.
A hit = the member's newest hidden title is in their top {K} picks. Picks (model and popular) exclude every title the member played in training, so rewatch targets cannot hit.

| | Hit Rate@10 |
|---|---|
| Two-tower model (PyTorch, {DIM}-d towers, {ROUNDS} rounds) | **{mh:.1%}** ({round(mh*n)}/{n}) |
| Popular top 10 (most-played titles the member has not played) | {ph:.1%} ({round(ph*n)}/{n}) |

Reported round: {ROUNDS} (final). Per round on the hidden week: """+", ".join(f"r{x['round']} {x['holdout_hit10_model']:.1%}" for x in rounds)+f"""

Title tower input = catalog description (TF-IDF) + type/genre/maturity, so zero-play titles get vectors: {', '.join(zero) or 'none'}
(TT-0300 vector norm {np.linalg.norm(V[ix['TT-0300']]):.2f}; appears in {in_picks} members' top 10).
Fixed hyperparameters, not tuned on the hidden week; a 3-9 Nov validation week gave model {vh:.1%} vs popular {vp:.1%}.
W&B run: {url}
Files: model/twotower.pt (weights), model/twotower_vectors.npz (member_vectors, title_vectors, ids), model/title_features.npy.
""");print('HOLDOUT',round(mh,3),round(ph,3),url)
