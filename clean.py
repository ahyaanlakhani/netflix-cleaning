"""Clean Netflix viewing events, match to catalog, split at cutoff.
Run: python clean.py   (stdlib only)"""
import csv,collections as C
CUTOFF="2026-11-10"            # final week = 10-16 Nov; held out from training
FILM_MAX,SER_MAX,DONE=2.0,10.0,0.90
rd=lambda p:list(csv.DictReader(open(p,encoding='utf-8-sig')))
cat={r['title_id']:r for r in rd('catalog.csv')};ev=rd('events_raw.csv')
keep=[];drop=[]
for r in ev:
    t=cat.get(r['title_id']);why=None
    if r['watch_seconds']=='':why='watch time missing'
    elif not t:why='title_id not in catalog'
    elif not t['runtime_minutes']:why='catalog runtime missing, duration unverifiable'
    elif r['event_datetime']<t['release_date']:why='viewed before title release date'
    else:
        p=int(r['watch_seconds'])/60/int(t['runtime_minutes'])
        if p>(FILM_MAX if t['title_type']=='film' else SER_MAX):why='implausibly long watch time'
    if why:drop.append({**r,'reason':why});continue
    p=int(r['watch_seconds'])/60/int(t['runtime_minutes'])
    ok=p>=DONE
    keep.append({**r,'title_name':t['title_name'],'title_type':t['title_type'],'primary_genre':t['primary_genre'],
      'runtime_minutes':t['runtime_minutes'],'pct_of_runtime':round(p,4),
      'completed':str(ok),'completed_original':r['completed'],'completed_corrected':str(ok!=(r['completed']=='True')),
      'device_type':r['device_type'] or 'unknown','country':r['country'] or 'unknown',
      'kids_on_mature_title':str(r['profile_type']=='kids' and t['maturity_rating'] in('R','TV-MA'))})
keep.sort(key=lambda r:(r['event_datetime'],r['event_id']))
def w(p,rows):
    with open(p,'w',newline='',encoding='utf-8') as f:
        x=csv.DictWriter(f,list(rows[0]));x.writeheader();x.writerows(rows)
w('events_train.csv',[r for r in keep if r['event_datetime']<CUTOFF])
w('events_holdout.csv',[r for r in keep if r['event_datetime']>=CUTOFF])
w('events_dropped.csv',drop)
print(len(ev),len(keep),C.Counter(d['reason'] for d in drop))
