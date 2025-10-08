import os
import io
import warnings
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

warnings.filterwarnings("ignore")

# ---------- ページ/フォント設定 ----------
st.set_page_config(page_title="C3slim データ分析アプリ", page_icon="📊", layout="wide")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS", "Yu Gothic", "Meiryo"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- ユーティリティ ----------
def numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def special_mask(series: pd.Series) -> pd.Series:
    """'None'(大小無視)と空白行を '欠損として数えない' ために保持するブールマスク"""
    s = series.astype(str)
    return s.str.lower().eq("none") | s.str.strip().eq("")

@st.cache_data(show_spinner=False)
def load_excel(file) -> tuple[dict, dict]:
    """全シート読込 + 2行目(インデックス1)を基準値候補として保持、3行目以降で最初の非空行からデータ開始"""
    xl = pd.ExcelFile(file)
    all_sheets, baseline_values = {}, {}
    for sheet in xl.sheet_names:
        df_full = xl.parse(sheet_name=sheet, header=0)
        baseline_values[sheet] = {} if len(df_full) < 2 else df_full.iloc[1].to_dict()
        # データ開始行（2行目=インデックス2以降の最初の非NaN行）
        start = 2
        for idx in range(2, len(df_full)):
            if not df_full.iloc[idx].isna().all():
                start = idx
                break
        all_sheets[sheet] = df_full.iloc[start:].reset_index(drop=True)
    return all_sheets, baseline_values

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """不要行除去 + 数値化（カンマ除去）。'None'や空白は欠損扱いしないため列ごとにマスク保持。"""
    dfc = df.dropna(how="all").reset_index(drop=True).copy()
    masks = {}
    for c in dfc.columns:
        if dfc[c].dtype == "object":
            m = special_mask(dfc[c])
            masks[c] = m
            # 数値化の判定（変換成功率 > 50%）
            temp = pd.to_numeric(dfc[c].astype(str).str.replace(",", ""), errors="coerce")
            if temp.notna().mean() > 0.5:
                dfc[c] = temp
        else:
            masks[c] = pd.Series(False, index=dfc.index)
    return dfc, masks

def basic_stats(df: pd.DataFrame, masks: dict, baseline: dict | None = None) -> pd.DataFrame | None:
    cols = numeric_cols(df)
    if not cols or len(df) == 0:
        return None
    rows = []
    n = len(df)
    for c in cols:
        data = df[c]
        miss = (data.isna() & ~masks.get(c, pd.Series(False, index=df.index))).sum()
        base = baseline.get(c, "") if baseline else ""
        try:
            if base != "" and pd.notna(base):
                base = round(float(base), 2)
        except Exception:
            pass
        cnt = data.count()
        rows.append({
            "列名": c,
            "基準値": base,
            "データ数": int(cnt),
            "平均値": round(data.mean(), 2) if cnt else np.nan,
            "標準偏差": round(data.std(ddof=1), 2) if cnt else np.nan,
            "最小値": data.min() if cnt else np.nan,
            "最大値": data.max() if cnt else np.nan,
            "欠損数": int(miss),
            "欠損率(%)": round(miss / n * 100, 2)
        })
    return pd.DataFrame(rows)

def calc_cpk(df: pd.DataFrame, usl: float, lsl: float) -> pd.DataFrame:
    out = []
    for c in numeric_cols(df):
        s = df[c].dropna()
        if len(s) <= 1:
            out.append({"列名": c, "平均値": "N/A", "標準偏差": "N/A", "Cp": "計算不可", "Cpk": "計算不可"})
            continue
        mu, sd = s.mean(), s.std(ddof=1)
        if sd == 0:
            out.append({"列名": c, "平均値": round(mu, 2), "標準偏差": 0, "Cp": "計算不可", "Cpk": "計算不可"})
            continue
        cp = (usl - lsl) / (6 * sd)
        cpk = min((usl - mu) / (3 * sd), (mu - lsl) / (3 * sd))
        out.append({"列名": c, "平均値": round(mu, 2), "標準偏差": round(sd, 2), "Cp": round(cp, 3), "Cpk": round(cpk, 3)})
    return pd.DataFrame(out)

def hist_with_gauss(df: pd.DataFrame, col: str, usl: float | None, lsl: float | None):
    data = df[col].dropna()
    if data.empty:
        return None
    mu, sd = data.mean(), data.std(ddof=1)
    hist_values, bins = np.histogram(data, bins=30)
    centers = (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins)

    fig = go.Figure()
    fig.add_bar(x=centers, y=hist_values, width=widths, name="頻度", opacity=0.7)

    if sd > 0:
        xs = np.linspace(data.min(), data.max(), 200)
        pdf = stats.norm.pdf(xs, mu, sd)
        fig.add_scatter(x=xs, y=pdf, name="ガウス分布", mode="lines")
        yaxis2 = dict(overlaying="y", side="right", title="確率密度")
        fig.update_layout(yaxis2=yaxis2)

    title = f"{col} のヒストグラム" + ("" if sd > 0 else "（全データ同一）")
    fig.update_layout(title=title, xaxis_title=col, yaxis_title="頻度", hovermode="x unified", legend=dict(x=0.7, y=1))

    if usl is not None:
        fig.add_vline(x=usl, line_dash="dash", line_color="red", line_width=2, annotation_text="USL", annotation_position="top")
    if lsl is not None:
        fig.add_vline(x=lsl, line_dash="dash", line_color="orange", line_width=2, annotation_text="LSL", annotation_position="top")
    return fig

def generate_pdf(df_clean: pd.DataFrame, stats_df: pd.DataFrame | None, sheet_name: str, masks: dict) -> bytes | None:
    try:
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = tmp_pdf.name
        tmp_pdf.close()
        tmp_imgs = []

        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story, styles = [], getSampleStyleSheet()

        title = ParagraphStyle("T", parent=styles["Heading1"], fontSize=20, textColor=colors.HexColor("#1f77b4"), alignment=1, spaceAfter=18)
        story += [
            Paragraph(f"Excel Data Analysis Report<br/>{sheet_name}", title),
            Paragraph(datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S"), styles["Normal"]),
            Spacer(1, 0.3 * inch),
        ]

        # 概要
        total_missing = 0
        for c in df_clean.columns:
            m = masks.get(c, pd.Series(False, index=df_clean.index))
            total_missing += int((df_clean[c].isna() & ~m).sum())

        overview = [["Metric", "Value"],
                    ["Total Rows", str(len(df_clean))],
                    ["Total Columns", str(df_clean.shape[1])],
                    ["Numeric Columns", str(len(numeric_cols(df_clean)))],
                    ["Missing Values", str(total_missing)]]
        t = Table(overview, colWidths=[3 * inch, 2 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        story += [Paragraph("Data Overview", styles["Heading2"]), Spacer(1, 0.1 * inch), t, Spacer(1, 0.2 * inch)]

        # 基本統計量
        if stats_df is not None and not stats_df.empty:
            head_df = stats_df.head(10)
            data = [head_df.columns.tolist()] + [[str(v) for v in r] for r in head_df.to_numpy()]
            t2 = Table(data, repeatRows=1)
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            story += [Paragraph("Basic Statistics", styles["Heading2"]), Spacer(1, 0.1 * inch), t2, Spacer(1, 0.2 * inch)]

        # ヒスト（最初の6数値列）
        ncols = numeric_cols(df_clean)[:6]
        if ncols:
            story += [PageBreak(), Paragraph("Distribution Charts", styles["Heading2"]), Spacer(1, 0.1 * inch)]
            for i, c in enumerate(ncols):
                fig, ax = plt.subplots(figsize=(6, 3))
                df_clean[c].dropna().hist(bins=30, ax=ax, edgecolor="black")
                ax.set_title(f"{c} Distribution")
                ax.set_xlabel(c); ax.set_ylabel("Frequency")

                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                plt.savefig(tmp_img.name, bbox_inches="tight", dpi=110)
                plt.close()
                tmp_imgs.append(tmp_img.name)

                story += [Image(tmp_img.name, width=5 * inch, height=2.5 * inch), Spacer(1, 0.2 * inch)]
                if (i + 1) % 2 == 0 and i < len(ncols) - 1:
                    story.append(PageBreak())

        doc.build(story)

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        os.unlink(pdf_path)
        for p in tmp_imgs:
            try: os.unlink(p)
            except: pass

        return pdf_bytes
    except Exception as e:
        st.error(f"PDF生成エラー: {e}")
        return None

# ---------- メイン ----------
def main():
    st.title("📊 C3slim データ分析アプリ")
    st.markdown("---")

    # サイドバー：アップロード & シート選択
    st.sidebar.header("📁 ファイルアップロード")
    up = st.sidebar.file_uploader("Excelファイルを選択 (.xlsx/.xls)", type=["xlsx", "xls"])

    if "all_sheets" not in st.session_state: st.session_state.all_sheets = {}
    if "baseline" not in st.session_state: st.session_state.baseline = {}
    if "sheet" not in st.session_state: st.session_state.sheet = None

    if up:
        with st.spinner("ファイルを読み込み中..."):
            sheets, base = load_excel(up)
            st.session_state.all_sheets = sheets
            st.session_state.baseline = base
            if st.session_state.sheet not in sheets:
                st.session_state.sheet = list(sheets.keys())[0]
        st.sidebar.success(f"✅ 読込成功（シート数：{len(sheets)}）")

        st.sidebar.subheader("📑 シート選択")
        sel = st.sidebar.selectbox("分析するシート", list(st.session_state.all_sheets.keys()),
                                   index=list(st.session_state.all_sheets.keys()).index(st.session_state.sheet))
        if sel != st.session_state.sheet:
            st.session_state.sheet = sel
            st.rerun()

    if not st.session_state.get("sheet"):
        st.info("👈 左のサイドバーからExcelファイルをアップロードしてください。")
        st.markdown("""
        **使い方**  
        1) ファイルをアップロード → 2) データ確認 → 3) 基本統計/CpCpk → 4) グラフ → 5) フィルタ → 6) エクスポート
        """)
        return

    # 対象データ
    df_raw = st.session_state.all_sheets[st.session_state.sheet]
    df_clean, masks = clean_data(df_raw)
    base = st.session_state.baseline.get(st.session_state.sheet, {})

    # ---------- タブ ----------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📋 データプレビュー", "📈 基本統計量", "📊 ヒストグラム", "🔵 散布図", "🎨 カスタムグラフ", "🔍 データフィルタ"])

    with tab1:
        st.subheader("元データ / クリーニング後")
        c1, c2 = st.columns(2)
        with c1:
            st.write("🔍 **元データ**"); st.dataframe(df_raw.head(10), use_container_width=True)
            st.caption(f"表示: 上位10行 / 全{len(df_raw)}行")
        with c2:
            st.write("🧹 **クリーニング後**"); st.dataframe(df_clean.head(10), use_container_width=True)
            st.caption(f"表示: 上位10行 / 全{len(df_clean)}行")

        st.subheader("📋 データ型/欠損")
        c1, c2 = st.columns(2)
        with c1:
            info = pd.DataFrame({"列名": df_raw.columns, "データ型": [str(t) for t in df_raw.dtypes], "欠損値": df_raw.isna().sum().to_list()})
            st.dataframe(info, use_container_width=True)
        with c2:
            adj_missing = []
            for c in df_clean.columns:
                m = masks.get(c, pd.Series(False, index=df_clean.index))
                adj_missing.append(int((df_clean[c].isna() & ~m).sum()))
            info2 = pd.DataFrame({"列名": df_clean.columns, "データ型": [str(t) for t in df_clean.dtypes], "欠損値": adj_missing})
            st.dataframe(info2, use_container_width=True)

    with tab2:
        st.subheader("📈 基本統計量")
        stats_df = basic_stats(df_clean, masks, base)
        if stats_df is not None:
            st.dataframe(stats_df, use_container_width=True)
            st.download_button("📥 統計量CSV", stats_df.to_csv(index=False, encoding="utf-8-sig"), "basic_statistics.csv", "text/csv")
            st.markdown("---")

            st.subheader("🎯 工程能力指数 (Cp, Cpk)")
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1: usl = st.number_input("規格上限値 (USL)", value=0.0, step=0.1, format="%.3f")
            with c2: lsl = st.number_input("規格下限値 (LSL)", value=0.0, step=0.1, format="%.3f")
            with c3:
                st.write("")
                do = st.button("計算する", type="primary")
            if do:
                if usl <= lsl:
                    st.error("USL は LSL より大きい値を入力してください。")
                else:
                    cpk_df = calc_cpk(df_clean, usl, lsl)
                    st.dataframe(cpk_df, use_container_width=True)
                    st.info("目安：1.67≧優良 / 1.33≧良好 / 1.00≧許容 / 1.00未満は改善要")
                    st.download_button("📥 Cp/Cpk CSV", cpk_df.to_csv(index=False, encoding="utf-8-sig"),
                                       "process_capability.csv", "text/csv")
        else:
            st.warning("数値列が見つかりません。")

    with tab3:
        st.subheader("📊 ヒストグラム")
        ncols = numeric_cols(df_clean)
        if not ncols:
            st.warning("数値列が見つかりません。")
        else:
            col = st.selectbox("列を選択", ncols)
            # 基準値提示とコピー
            base_val = base.get(col, "")
            if base_val != "" and pd.notna(base_val):
                try:
                    st.info(f"💡 基準値: {float(base_val):.3f}（USL/LSLへ転記可）")
                except Exception:
                    st.info(f"💡 基準値: {base_val}")

            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                usl_text = st.text_input(f"{col} の USL", value="", key=f"usl_{col}")
            with c2:
                lsl_text = st.text_input(f"{col} の LSL", value="", key=f"lsl_{col}")
            with c3:
                if base_val != "" and pd.notna(base_val):
                    st.write("")
                    if st.button("📋 基準値を転記", key=f"copy_{col}"):
                        try:
                            b = str(float(base_val))
                            st.session_state[f"usl_{col}"] = b
                            st.session_state[f"lsl_{col}"] = b
                            st.rerun()
                        except Exception:
                            st.error("基準値を数値に変換できません。")

            def to_float_or_none(s: str):
                s = (s or "").strip()
                try: return float(s) if s else None
                except: return None

            usl_v, lsl_v = to_float_or_none(usl_text), to_float_or_none(lsl_text)
            if usl_v is not None and lsl_v is not None and usl_v <= lsl_v:
                st.warning("USL は LSL より大きい値にしてください。")

            fig = hist_with_gauss(df_clean, col, usl_v, lsl_v)
            if fig: st.plotly_chart(fig, use_container_width=True)

            # サマリー & Cp/Cpk（当該列のみ）
            s = df_clean[col].dropna()
            if not s.empty:
                st.markdown("#### 📊 統計サマリー")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("データ数", len(s))
                c2.metric("平均値", f"{s.mean():.3f}")
                c3.metric("標準偏差", f"{s.std(ddof=1):.3f}")
                c4.metric("最小値", f"{s.min():.3f}")
                c5.metric("最大値", f"{s.max():.3f}")

                if usl_v is not None and lsl_v is not None and usl_v > lsl_v:
                    sd = s.std(ddof=1)
                    if sd > 0:
                        cp = (usl_v - lsl_v) / (6 * sd)
                        cpk = min((usl_v - s.mean()) / (3 * sd), (s.mean() - lsl_v) / (3 * sd))
                        st.markdown("#### 🎯 工程能力指数")
                        d1, d2 = st.columns(2)
                        d1.metric("Cp", f"{cp:.3f}")
                        d2.metric("Cpk", f"{cpk:.3f}")
                        if cpk >= 1.67: st.success("✅ 優良")
                        elif cpk >= 1.33: st.info("👍 良好")
                        elif cpk >= 1.00: st.warning("⚠️ 許容")
                        else: st.error("❌ 改善が必要")
                    else:
                        st.warning("標準偏差が0のため計算できません。")

    with tab4:
        st.subheader("🔵 散布図")
        ncols = numeric_cols(df_clean)
        if len(ncols) < 2:
            st.warning("少なくとも2つの数値列が必要です。")
        else:
            c1, c2 = st.columns(2)
            with c1: x = st.selectbox("X軸", ncols, key="scatter_x")
            with c2: y = st.selectbox("Y軸", [c for c in ncols if c != x], key="scatter_y")
            color_col = st.selectbox("色分け列（任意）", ["なし"] + df_clean.columns.tolist(), key="scatter_color")
            if st.button("📈 作成", type="primary"):
                color_param = None if color_col == "なし" else color_col
                fig = px.scatter(df_clean, x=x, y=y, color=color_param, title=f"{y} vs {x}")
                fig.update_layout(title_font_size=16, xaxis_title=x, yaxis_title=y)
                st.plotly_chart(fig, use_container_width=True)
                corr = df_clean[[x, y]].corr().iloc[0, 1]
                st.metric("相関係数", f"{corr:.3f}")
                st.info("✅ 強い相関" if abs(corr) >= 0.7 else ("👍 中程度の相関" if abs(corr) >= 0.4 else "⚠️ 弱い相関"))

    with tab5:
        st.subheader("🎨 カスタムグラフ")
        ncols = numeric_cols(df_clean)
        if not ncols:
            st.warning("数値列が見つかりません。")
        else:
            gtype = st.selectbox("種類", ["散布図", "折れ線", "棒", "面"])
            all_cols = df_clean.columns.tolist()
            if gtype == "散布図":
                x = st.selectbox("X軸", ncols, key="gx")
                y = st.selectbox("Y軸", [c for c in ncols if c != x], key="gy")
            else:
                x = st.selectbox("X軸", all_cols, key="gx2")
                y = st.selectbox("Y軸", ncols, key="gy2")
            color_col = st.selectbox("色分け列（任意）", ["なし"] + all_cols, key="gcolor")
            if st.button("📈 生成", type="primary"):
                color_param = None if color_col == "なし" else color_col
                if gtype == "散布図":
                    fig = px.scatter(df_clean, x=x, y=y, color=color_param, title=f"{y} vs {x}")
                elif gtype == "折れ線":
                    fig = px.line(df_clean, x=x, y=y, color=color_param, title=f"{y} の推移")
                elif gtype == "棒":
                    fig = px.bar(df_clean, x=x, y=y, color=color_param, title=f"{y} の棒グラフ")
                else:
                    fig = px.area(df_clean, x=x, y=y, color=color_param, title=f"{y} の面グラフ")
                fig.update_layout(title_font_size=16, xaxis_title=x, yaxis_title=y)
                st.plotly_chart(fig, use_container_width=True)

                # 参考：平均
                c1, c2 = st.columns(2)
                with c1:
                    xv = df_clean[x]
                    xv_mean = f"{xv.mean():.2f}" if pd.api.types.is_numeric_dtype(xv) else "N/A"
                    st.metric(f"{x} 平均", xv_mean)
                with c2:
                    st.metric(f"{y} 平均", f"{df_clean[y].mean():.2f}")

    with tab6:
        st.subheader("🔍 データフィルタ")
        col = st.selectbox("フィルタ列", df_clean.columns.tolist())
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            vmin, vmax = float(df_clean[col].min()), float(df_clean[col].max())
            lo, hi = st.slider(f"{col} 範囲", min_value=vmin, max_value=vmax, value=(vmin, vmax))
            filt = df_clean[(df_clean[col] >= lo) & (df_clean[col] <= hi)]
        else:
            vals = df_clean[col].dropna().unique().tolist()
            chosen = st.multiselect(f"{col} 値", vals, default=vals)
            filt = df_clean[df_clean[col].isin(chosen)]
        st.info(f"💡 元: {len(df_clean)}行 → フィルタ後: {len(filt)}行")
        st.dataframe(filt, use_container_width=True)

        if len(filt) > 0:
            st.markdown("#### 📈 フィルタ後 基本統計量")
            fstats = basic_stats(filt, masks, base)
            if fstats is not None:
                st.dataframe(fstats, use_container_width=True)
                st.download_button("📥 フィルタ後CSV", filt.to_csv(index=False, encoding="utf-8-sig"),
                                   "filtered_data.csv", "text/csv")

    # ---------- エクスポート ----------
    st.markdown("---")
    st.subheader("💾 エクスポート")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("📥 元データCSV", df_raw.to_csv(index=False, encoding="utf-8-sig"),
                           "original_data.csv", "text/csv")
    with c2:
        st.download_button("📥 クリーニング後CSV", df_clean.to_csv(index=False, encoding="utf-8-sig"),
                           "cleaned_data.csv", "text/csv")
    with c3:
        if st.button("📄 PDFレポート生成", type="primary"):
            with st.spinner("PDFを生成中..."):
                stats_df = basic_stats(df_clean, masks, base)
                pdf = generate_pdf(df_clean, stats_df, st.session_state.sheet or "データ", masks)
                if pdf:
                    st.download_button("📥 PDFレポート", pdf,
                                       f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                       "application/pdf")

if __name__ == "__main__":
    main()
