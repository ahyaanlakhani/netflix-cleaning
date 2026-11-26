import json,jsonschema,csv
from fastapi.testclient import TestClient
from api import app,member_vec,played
S=json.load(open('recommendation_response_schema.json'));c=TestClient(app)
ok=jsonschema.Draft202012Validator(S)
def err_ok(b):jsonschema.validate(b,{'$ref':'#/$defs/ErrorResponse','$defs':S['$defs']})
def test_every_member_valid_unique_sorted_and_unseen():
    for m in member_vec:
        r=c.get(f'/recommendations/{m}');assert r.status_code==200,m;b=r.json();ok.validate(b)
        ids=[x['title_id'] for x in b['recommendations']];sc=[x['score'] for x in b['recommendations']]
        assert len(set(ids))==10 and sc==sorted(sc,reverse=True) and b['user_id']==m
        assert not set(ids)&played.get(m,set()),m
def test_names_and_genres_match_catalog():
    cat={r['title_id']:r for r in csv.DictReader(open('catalog.csv',encoding='utf-8'))}
    for x in c.get('/recommendations/M-10042').json()['recommendations']:
        assert x['title_name']==cat[x['title_id']]['title_name'] and x['primary_genre']==cat[x['title_id']]['primary_genre']
def test_404_bodies():
    for u in['M-99999','not-an-id','M-1']:
        r=c.get(f'/recommendations/{u}');assert r.status_code==404;b=r.json();err_ok(b);assert b['user_id']==u
