"""GET /recommendations/{user_id} - follows recommendation_response_schema.json (v1.0) exactly.
Run: uvicorn api:app --port 8000   (needs Qdrant on QDRANT_URL, collection 'title_vectors')"""
import csv,os,time
from typing import Literal,Optional
import numpy as np
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel,ConfigDict,Field
from qdrant_client import QdrantClient
Genre=Literal["Drama","Comedy","Thriller","Crime","Documentary","Sci-Fi","Fantasy","Romance","Horror","Action","Animation","Reality"]
class RecommendationItem(BaseModel):
    model_config=ConfigDict(extra="forbid")
    title_id:str=Field(pattern=r"^TT-\d{4}$");title_name:str=Field(min_length=1);primary_genre:Genre;score:float
class RecommendationResponse(BaseModel):
    model_config=ConfigDict(extra="forbid")
    user_id:str=Field(pattern=r"^M-\d{5}$")
    recommendations:list[RecommendationItem]=Field(min_length=10,max_length=10)
    latency_ms:Optional[float]=Field(default=None,ge=0)
class ErrorResponse(BaseModel):
    model_config=ConfigDict(extra="forbid")
    detail:str=Field(min_length=1);user_id:str
K,COLL=10,"title_vectors"
rd=lambda p:list(csv.DictReader(open(p,encoding='utf-8')))
catalog={r['title_id']:r for r in rd('catalog.csv')}
z=np.load('model/twotower_vectors.npz');member_vec={str(m):v for m,v in zip(z['member_ids'],z['member_vectors'])}
played={}                                   # every title a member played in the TRAINING split
for r in rd('events_train.csv'):played.setdefault(r['member_id'],set()).add(r['title_id'])
qd=QdrantClient(url=os.environ.get("QDRANT_URL","http://localhost:6333"))
app=FastAPI(title="Top Picks")
def not_found(user_id,detail):return JSONResponse(status_code=404,content=ErrorResponse(detail=detail,user_id=user_id).model_dump())
@app.exception_handler(RequestValidationError)
async def _v(request,exc):return not_found(request.path_params.get("user_id",""),"Invalid request")
@app.get("/recommendations/{user_id}",response_model=RecommendationResponse,response_model_exclude_none=True,responses={404:{"model":ErrorResponse}})
def recommendations(user_id:str):
    t0=time.perf_counter();vec=member_vec.get(user_id)
    if vec is None:return not_found(user_id,f"No recommendations available for member {user_id}")
    seen=played.get(user_id,set())
    # ask for extra results so 10 remain after removing every title already played in training
    hits=qd.query_points(COLL,query=vec.tolist(),limit=min(len(catalog),K+len(seen)),with_payload=True).points
    picks,ids=[],set()
    for h in hits:
        tid=h.payload["title_id"]
        if tid in seen or tid in ids:continue
        ids.add(tid);c=catalog[tid];picks.append(RecommendationItem(title_id=tid,title_name=c['title_name'],primary_genre=c['primary_genre'],score=float(h.score)))
        if len(picks)==K:break
    if len(picks)<K:return not_found(user_id,f"Fewer than {K} unseen titles available for member {user_id}")
    return RecommendationResponse(user_id=user_id,recommendations=picks,latency_ms=round((time.perf_counter()-t0)*1000,3))
