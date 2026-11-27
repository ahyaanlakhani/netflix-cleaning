Build a Streamlit page called Member top picks in app.py. It has a text box for a member ID and a button. On click, call GET http://127.0.0.1:8000/recommendations/{user_id} with requests (address configurable through API_BASE; 127.0.0.1, not localhost, because localhost adds ~2 s on Windows).

Display: Show each of the 10 rows as Rank (1-10), Title (title_name), Genre (primary_genre), Title ID (title_id) and Score (score, rounded to 3 decimals), in the API's order.

Timing: Time only the requests.get(...) call, with time.perf_counter() on the line right before it and right after it, and show the result under the table as "API call took X ms (server built it in Y ms)", where Y is the latency_ms the API returns.

A 404 shows the API's detail message; an empty box shows a warning; an unreachable API shows an error.

Recently watched: above the picks, show the member's last 5 watched titles (newest first, with date, title name and genre) read from events_train.csv, or a note if there is no history.

Popular comparison: in a side column next to the picks, show the popular top 10 that every member sees today (the 10 most-played titles in events_train.csv: rank, title, genre, plays), so a reviewer can compare it with the member's picks.
