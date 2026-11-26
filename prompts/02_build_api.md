Build GET /recommendations/{user_id} in FastAPI that matches the response contract exactly, searches Qdrant, and removes every title the member played in the training split.

Follow recommendation_response_schema.json exactly for every 200 and 404 body.

Ask Qdrant for extra results (10 + number of titles the member played) so 10 titles remain after removing played ones.

Save two real API responses as JSON files in the repo (examples/).
