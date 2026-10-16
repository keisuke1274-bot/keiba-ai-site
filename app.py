
import re
import requests
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from pathlib import Path
from bs4 import BeautifulSoup

st.set_page_config(page_title="競馬AI予想", page_icon="🏇", layout="wide")
st.title("🏇 競馬AI予想サイト")
st.caption("既存AIモデル使用｜出馬表・過去成績から予測")

MODEL_PATH = Path(__file__).parent / "horse_model_with_suitability.joblib"
model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 "
        "Mobile/15E148 Safari/604.1"
    )
}

# DBのコード体系をもとにした暫定対応。学習データとの意味の一致は未検証。
GOING_CODES = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不良": 3}
WEATHER_CODES = {"晴": 0, "曇": 1, "雨": 2, "小雨": 3, "小雪": 4, "雪": 5}

def get_soup(url):
    response = requests.get(url, headers=HEADERS, timeout=25)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    return BeautifulSoup(response.text, "html.parser")

def race_info(soup, race_id):
    data = soup.select_one(".RaceData01")
    text = data.get_text(" ", strip=True) if data else soup.get_text(" ", strip=True)
    m = re.search(r"(芝|ダート|ダ|障害|障)\s*([0-9,]+)\s*m", text)
    if not m:
        raise ValueError("レースの距離・コースを取得できませんでした。")
    surface_text = m.group(1)
    surface = 1 if surface_text == "芝" else 0
    distance = int(m.group(2).replace(",", ""))

    going_match = re.search(r"馬場\s*[:：]\s*(稍重|不良|稍|重|良)", text)
    weather_match = re.search(r"天候\s*[:：]\s*(小雨|小雪|晴|曇|雨|雪)", text)
    if not going_match or not weather_match:
        raise ValueError("馬場状態または天候を取得できませんでした。")
    going = going_match.group(1)
    weather = weather_match.group(1)
    if going not in GOING_CODES or weather not in WEATHER_CODES:
        raise ValueError("馬場・天候のコードが未対応です。")
    place_id = int(race_id[4:6])
    if not 1 <= place_id <= 10:
        raise ValueError("競馬場コードを判定できません。")
    return {
        "distance": distance,
        "course": surface,
        "course_sit": GOING_CODES[going],
        "weather": WEATHER_CODES[weather],
        "place_id": place_id,
    }

def get_entries(soup):
    table = soup.select_one("table.Shutuba_Table")
    if table is None:
        tables = soup.find_all("table")
        table = next(
            (t for t in tables if t.select_one('a[href*="/horse/"]')),
            None
        )
    if table is None:
        raise ValueError("出走馬一覧を見つけられませんでした。")
    rows = []
    seen = set()
    for tr in table.select("tr"):
        link = tr.select_one('a[href*="/horse/"]')
        if not link:
            continue
        match = re.search(r"/horse/(\d+)", link.get("href", ""))
        if not match:
            continue
        hid = match.group(1)
        if hid in seen:
            continue
        seen.add(hid)
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        row_text = " ".join(cells)
        horse_no = None
        for cls in ["Umaban", "Umaban01"]:
            el = tr.select_one("." + cls)
            if el:
                found = re.search(r"\d+", el.get_text(" ", strip=True))
                if found:
                    horse_no = int(found.group())
                    break
        if horse_no is None:
            # 一般的な出馬表のセルから馬番を推定
            for cell in tr.find_all("td"):
                cls = " ".join(cell.get("class", []))
                if "Umaban" in cls:
                    found = re.search(r"\d+", cell.get_text(" ", strip=True))
                    if found:
                        horse_no = int(found.group())
                        break
        name = link.get_text(" ", strip=True)
        weight = np.nan
        weight_diff = np.nan
        for cell in tr.find_all(["td", "th"]):
            cls = " ".join(cell.get("class", []))
            txt = cell.get_text(" ", strip=True)
            if "Weight" in cls or re.fullmatch(r"\d{3}\([+-]?\d+\)", txt):
                wm = re.search(r"(\d{3})\s*\(([+-]?\d+)\)", txt)
                if wm:
                    weight = float(wm.group(1))
                    weight_diff = float(wm.group(2))
        # 斤量はセルの見出しから対応付け
        carried = np.nan
        headers = [th.get_text(" ", strip=True) for th in table.select("tr th")]
        row_cells = tr.find_all(["th", "td"])
        for i, h in enumerate(headers):
            if "斤量" in h and i < len(row_cells):
                wm = re.search(r"\d{2}(?:\.\d)?", row_cells[i].get_text(" ", strip=True))
                if wm:
                    carried = float(wm.group())
                    break
        if horse_no is None:
            # 先頭付近の短い数値セルを馬番候補として探す
            for cell in row_cells[:4]:
                txt = cell.get_text(" ", strip=True)
                if txt.isdigit() and 1 <= int(txt) <= 18:
                    horse_no = int(txt)
        rows.append({
            "horse_id": hid, "馬番": horse_no, "馬名": name,
            "斤量": carried, "馬体重": weight, "馬体重増減": weight_diff,
            "出馬表情報": row_text
        })
    if not rows:
        raise ValueError("出走馬を抽出できませんでした。")
    return rows

def horse_history(hid):
    soup = get_soup(f"https://db.netkeiba.com/horse/{hid}/")
    table = soup.select_one("table.db_h_race_results")
    if table is None:
        table = next(
            (t for t in soup.find_all("table")
             if "着順" in t.get_text() and "距離" in t.get_text()),
            None
        )
    if table is None:
        raise ValueError(f"馬ID {hid} の過去成績表を取得できませんでした。")

    trs = table.select("tr")
    if len(trs) < 2:
        return []
    headers = [x.get_text(" ", strip=True) for x in trs[0].find_all(["th", "td"])]
    def col_index(words):
        for i, h in enumerate(headers):
            if any(w in h for w in words):
                return i
        return None

    rank_i = col_index(["着順"])
    dist_i = col_index(["距離"])
    date_i = col_index(["日付"])
    if rank_i is None or dist_i is None:
        raise ValueError(f"馬ID {hid} の成績表の列を判定できません。")

    records = []
    for tr in trs[1:]:
        cells = [x.get_text(" ", strip=True) for x in tr.find_all(["th", "td"])]
        if len(cells) <= max(rank_i, dist_i):
            continue
        rank_match = re.match(r"\s*(\d+)", cells[rank_i])
        if not rank_match:
            continue
        dm = re.search(r"(芝|ダート|ダ|障害|障)\s*([0-9,]+)", cells[dist_i])
        if not dm:
            continue
        surface = 1 if dm.group(1) == "芝" else 0
        date = cells[date_i] if date_i is not None and date_i < len(cells) else ""
        records.append({
            "rank": int(rank_match.group(1)),
            "surface": surface,
            "distance": int(dm.group(2).replace(",", "")),
            "date": date
        })
    return records

def build_features(entry, info):
    history = horse_history(entry["horse_id"])
    if not history:
        raise ValueError(f"{entry['馬名']} の過去成績が見つかりません。")

    # netkeibaの過去成績表は通常新しい順。日付が解釈できる場合は明示的に並べる。
    hist = pd.DataFrame(history)
    if "date" in hist.columns:
        parsed = pd.to_datetime(hist["date"], errors="coerce")
        if parsed.notna().any():
            hist = hist.assign(_date=parsed).sort_values("_date", ascending=False)

    last3 = hist.head(3)
    same_surface = hist[hist["surface"] == info["course"]]
    similar_distance = hist[
        (hist["surface"] == info["course"]) &
        ((hist["distance"] - info["distance"]).abs() <= 200)
    ]
    f = {
        "distance": info["distance"],
        "gate": entry["馬番"] if entry["馬番"] is not None else np.nan,
        "carried_weight": entry["斤量"],
        "body_weight": entry["馬体重"],
        "body_weight_diff": entry["馬体重増減"],
        "past_starts": len(hist),
        "last_rank": hist.iloc[0]["rank"],
        "avg_rank_last3": float(last3["rank"].mean()),
        "win_rate_last3": float((last3["rank"] == 1).mean()),
        "past_same_surface_starts": len(same_surface),
        "past_same_surface_win_rate": float((same_surface["rank"] == 1).mean()) if len(same_surface) else 0.0,
        "past_similar_distance_starts": len(similar_distance),
        "past_similar_distance_win_rate": float((similar_distance["rank"] == 1).mean()) if len(similar_distance) else 0.0,
        "place_id": info["place_id"],
        "course": info["course"],
        "course_sit": info["course_sit"],
        "weather": info["weather"],
    }
    return {k: f[k] for k in FEATURES}

st.success("既存AIモデルを読み込みました")
st.caption(f"モデル特徴量：{len(FEATURES)}項目")

st.warning(
    "コース・馬場・天候コードはDBの値をもとにした暫定対応です。"
    "学習時のコードとの完全一致は未検証です。スコアは校正済み勝率ではありません。"
)

race_id = st.text_input(
    "netkeibaのレースID（12桁）",
    placeholder="例：202605020411"
)

if st.button("出走馬を取得してAI予測", type="primary"):
    race_id = race_id.strip()
    if not re.fullmatch(r"\d{12}", race_id):
        st.error("12桁のレースIDを入力してください。")
    else:
        try:
            with st.spinner("出馬表とレース条件を取得中..."):
                url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
                soup = get_soup(url)
                info = race_info(soup, race_id)
                entries = get_entries(soup)

            st.write(
                f"条件：{info['distance']}m｜競馬場コード {info['place_id']}｜"
                f"出走馬 {len(entries)}頭"
            )
            results = []
            failures = []
            progress = st.progress(0)
            status = st.empty()

            for i, entry in enumerate(entries):
                status.write(f"過去成績を取得中：{i+1}/{len(entries)}　{entry['馬名']}")
                try:
                    features = build_features(entry, info)
                    X = pd.DataFrame([features], columns=FEATURES)
                    score = float(model.predict_proba(X)[0, 1])
                    results.append({
                        "馬番": entry["馬番"],
                        "馬名": entry["馬名"],
                        "AI予測スコア": score,
                        "過去走数": features["past_starts"],
                    })
                except Exception as e:
                    failures.append(f"{entry['馬名']}: {e}")
                progress.progress((i + 1) / len(entries))

            status.empty()
            progress.empty()

            if failures:
                st.warning(
                    f"{len(failures)}頭は過去成績を取得・変換できず、"
                    "予測順位から除外しました。"
                )
                for message in failures:
                    st.write("・" + message)

            if results:
                result_df = pd.DataFrame(results).sort_values(
                    "AI予測スコア", ascending=False
                ).reset_index(drop=True)
                result_df.insert(0, "予測順位", range(1, len(result_df) + 1))
                st.subheader("AI予測順位")
                st.dataframe(
                    result_df.style.format({"AI予測スコア": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(
                    "スコアはモデル出力値であり、実際の勝率を保証するものではありません。"
                )
                st.link_button("netkeibaの出馬表を開く", url)
            else:
                st.error("予測できる馬がいませんでした。上の取得エラーを確認してください。")

        except Exception as e:
            st.error(f"レース取得エラー：{e}")

st.divider()
with st.expander("モデル情報"):
    st.write("使用モデル：horse_model_with_suitability.joblib")
    st.write("使用特徴量：", FEATURES)
