# aiko

Streamlit-based "friendly store assistant" — natural-language search over a
local product CSV (DMart inventory) backed by OpenAI for chat and tool calls.

## Setup

```sh
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
```

Place your inventory CSV at `store_db/DMart.csv`. The app reads it on startup.

## Run

```sh
streamlit run app_base.py
```

`app_testing.py` is an experimental variant kept alongside for comparison.

## Notes

- Requires an OpenAI API key. The app validates the connection at startup
  and surfaces a clear error if it fails.
- DuckDuckGo search is enabled when `duckduckgo-search` is installed
  (already in `requirements.txt`).
- This repo and `chatbot/` are near-duplicates — `chatbot/` differs only in
  retaining a debug `print(query)` line. `aiko` is the canonical version.
