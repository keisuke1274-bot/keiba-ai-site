
import re
import requests
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from bs4 import BeautifulSoup

st.set_page_config(page_title="競馬AI予測", page_icon="🏇", layout="wide")
st.title("🏇 競馬AI予測サイト")

model_path = Path(__file__).parent / "horse_model_with_suitability.joblib"
model = joblib.load(model_path)
st.success("既存AIモデル接続済み")

st.subheader("レースを指定")
race_id = st.text_input("netkeibaのレースID（12桁）", placeholder="例：202605020411")

if st.button("出走馬を取得", type="primary"):
    race_id = race_id.strip()

    if not re.fullmatch(r"\d{12}", race_id):
        st.error("12桁のレースIDを入力してください")
    else:
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1"
        }

        try:
            with st.spinner("出馬表を取得中..."):
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

            entries = []
            seen = set()

            for table in soup.find_all("table"):
                for tr in table.select("tr"):
                    link = tr.select_one('a[href*="/horse/"]')
                    if link is None:
                        continue

                    match = re.search(r"/horse/(\d+)", link.get("href", ""))
                    if not match:
                        continue

                    hid = match.group(1)
                    if hid in seen:
                        continue
                    seen.add(hid)

                    cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                    entries.append({
                        "馬ID": hid,
                        "出馬表情報": " | ".join(cells)
                    })

            if entries:
                df = pd.DataFrame(entries)
                st.session_state["entries"] = df
                st.success(f"{len(df)}頭の馬IDを取得しました")
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.link_button("出馬表を開く", url)
            else:
                st.warning("出走馬を抽出できませんでした。ページ構造を確認する必要があります。")

        except Exception as e:
            st.error(f"取得エラー: {e}")

st.divider()
st.subheader("AIモデル情報")
st.write("使用特徴量:")
st.write(list(model.feature_names_in_))
st.info("現在は出走馬取得までの試作版です。過去成績の取得と特徴量変換を接続した後に予測を表示します。")
