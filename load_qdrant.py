"""Load the 300 title vectors into a local Qdrant collection (64-d, cosine), payload = title_id."""
import numpy as np
from qdrant_client import QdrantClient,models
z=np.load('model/twotower_vectors.npz');V,ids=z['title_vectors'],[str(t) for t in z['title_ids']]
assert V.shape==(300,64)
c=QdrantClient(url="http://localhost:6333");NAME="title_vectors"
if c.collection_exists(NAME):c.delete_collection(NAME)
c.create_collection(NAME,vectors_config=models.VectorParams(size=64,distance=models.Distance.COSINE))
c.upsert(NAME,points=[models.PointStruct(id=i,vector=V[i].tolist(),payload={"title_id":ids[i]}) for i in range(len(ids))],wait=True)
print("count:",c.count(NAME,exact=True).count)
p=c.scroll(NAME,scroll_filter=models.Filter(must=[models.FieldCondition(key="title_id",match=models.MatchValue(value="TT-0300"))]),with_vectors=True)[0]
print("TT-0300 present:",len(p)==1,"dim",len(p[0].vector))
