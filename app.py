"""Member top picks - Streamlit page that calls GET /recommendations/{user_id}.
Run API:  uvicorn api:app --port 8000     Run page:  streamlit run app.py   (from the repo folder)"""
import csv,os,time,collections as C
from pathlib import Path
import requests,streamlit as st
API=os.environ.get("API_BASE","http://127.0.0.1:8000")+"/recommendations/{user_id}"
HERE=Path(__file__).parent
@st.cache_data
def load():
    rd=lambda p:list(csv.DictReader(open(HERE/p,encoding='utf-8')))
    cat={r['title_id']:r for r in rd('catalog.csv')};ev=rd('events_train.csv')
    plays=C.Counter(r['title_id'] for r in ev)
    popular=[{"Rank":i,"Title":cat[t]['title_name'],"Genre":cat[t]['primary_genre'],"Plays":n} for i,(t,n) in enumerate(plays.most_common(10),1)]
    hist=C.defaultdict(list)
    for r in sorted(ev,key=lambda r:(r['event_datetime'],r['event_id'])):hist[r['member_id']].append(r)
    return cat,popular,hist
cat,popular,hist=load()
st.set_page_config(page_title="Member top picks",layout="wide")
st.title("Member top picks")
user_id=st.text_input("Member ID",placeholder="M-10042").strip()
if st.button("Get top picks"):
    if not user_id:
        st.warning("Enter a member ID.")
    else:
        try:
            t0=time.perf_counter()
            r=requests.get(API.format(user_id=user_id),timeout=10)
            call_ms=(time.perf_counter()-t0)*1000
        except requests.RequestException:
            st.error("Could not reach the recommendations API. Is it running?")
        else:
            if r.status_code==200:
                body=r.json();left,right=st.columns([3,2])
                with left:
                    recent=hist.get(body['user_id'],[])[-5:][::-1]
                    st.subheader("Recently watched")
                    if recent:
                        st.table([{"Date":e["event_datetime"],"Title":cat[e["title_id"]]["title_name"],"Genre":cat[e["title_id"]]["primary_genre"]} for e in recent])
                    else:st.caption("No viewing history in the training data.")
                    st.subheader(f"Top picks for {body['user_id']}")
                    st.table([{"Rank":i,"Title":x["title_name"],"Genre":x["primary_genre"],"Title ID":x["title_id"],"Score":round(x["score"],3)}
                              for i,x in enumerate(body["recommendations"],1)])
                    st.caption(f"API call took {call_ms:.0f} ms"+(f" (server built it in {body['latency_ms']:.0f} ms)" if body.get("latency_ms") is not None else ""))
                with right:
                    st.subheader("Popular top 10")
                    st.caption("The same list for every member, ranked by total plays before 10 Nov.")
                    st.table(popular)
            elif r.status_code==404:
                st.info(r.json().get("detail","No picks available for this member."))
            else:
                st.error(f"Unexpected response from the API (HTTP {r.status_code}).")
