"""Member top picks - Streamlit page that calls GET /recommendations/{user_id}.
Run API:  uvicorn api:app --port 8000     Run page:  streamlit run app.py"""
import requests,streamlit as st
import os,time
API=os.environ.get("API_BASE","http://127.0.0.1:8000")+"/recommendations/{user_id}"
st.set_page_config(page_title="Member top picks")
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
                body=r.json();st.subheader(f"Top picks for {body['user_id']}")
                st.table([{"Rank":i,"Title":x["title_name"],"Genre":x["primary_genre"],"Title ID":x["title_id"],"Score":round(x["score"],3)}
                          for i,x in enumerate(body["recommendations"],1)])
                st.caption(f"API call took {call_ms:.0f} ms"+(f" (server built it in {body['latency_ms']:.0f} ms)" if body.get("latency_ms") is not None else ""))
            elif r.status_code==404:
                st.info(r.json().get("detail","No picks available for this member."))
            else:
                st.error(f"Unexpected response from the API (HTTP {r.status_code}).")
