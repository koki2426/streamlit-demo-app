# app.py ― 日本語PDF対応・全件レポート・CSV文字化け対策
#            散布図：通常/累積件数/各値の件数/小計（ランニングサム）＆PDF出力対応・軸反転【完全版】
import warnings
from pathlib import Path
from datetime import datetime
import tempfile
import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

warnings.filterwarnings("ignore")
st.set_page_config(page_title="C3slim データ分析アプリ", page_icon="📊", layout="wide")

# ---------------- 日本語フォント登録（TTF優先→CIDフォールバック） ----------------
def register_japanese_fonts():
    """ReportLab/Matplotlib に日本語フォントを登録。
       1) fonts/NotoSansJP-*.ttf があればそれを埋め込み
       2) なければ CID フォント HeiseiKakuGo-W5 にフォールバック（PDFのみ日本語可）
    """
    app_dir = Path(__file__).parent
    fonts_dir = app_dir / "fonts"
    noto_reg = fonts_dir / "NotoSansJP-Regular.ttf"
    noto_bold = fonts_dir / "NotoSansJP-Bold.ttf"

    RL_FONT_REG = RL_FONT_BOLD = None
    MPL_FONT_NAME = None

    if noto_reg.exists():
        try:
            pdfmetrics.registerFont(TTFont("JP", str(noto_reg)))
            pdfmetrics.registerFont(TTFont("JP-Bold", str(noto_bold if noto_bold.exists() else noto_reg)))
            RL_FONT_REG, RL_FONT_BOLD = "JP", "JP-Bold"

            # Matplotlib にも同梱フォントを追加
            fm.fontManager.addfont(str(noto_reg))
            if noto_bold.exists():
                fm.fontManager.addfont(str(noto_bold))
            MPL_FONT_NAME = fm.FontProperties(fname=str(noto_reg)).get_name()
            plt.rcParams["font.family"] = MPL_FONT_NAME
            plt.rcParams["axes.unicode_minus"] = False
        except Exception as e:
            st.error(f"NotoSansJP の登録に失敗しました: {e}")

    # 同梱フォントが無い/失敗 → CID フォントへ（PDF内テキストはOK、matplotlib画像は環境既定フォント）
    if RL_FONT_REG is None:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))  # ゴシック体
            RL_FONT_REG = RL_FONT_BOLD = 'HeiseiKakuGo-W5'
            st.sidebar.warning("NotoSansJP が見つからないため CID フォント(HeiseiKakuGo-W5)でPDFを出力します。")
        except Exception as e:
            st.error(f"CIDフォント登録に失敗: {e}")

    # デバッグ表示
    st.sidebar.info(f"PDFフォント: {RL_FONT_REG} / 太字: {RL_FONT_BOLD} / MPL: {MPL_FONT_NAME or '未設定'}")
    return RL_FONT_REG, RL_FONT_BOLD, MPL_FONT_NAME

RL_FONT_REG, RL_FONT_BOLD, MPL_FONT_NAME = register_japanese_fonts()

# ---------------- ユーティリティ ----------------
def numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def special_mask(series: pd.Series) -> pd.Series:
    """'None'(大小無視) と 空白 を '欠損として数えない' 値として扱うためのマスク"""
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
    rows, n = [], len(df)
    for c in cols:
        data = df[c]
        miss = int((data.isna() & ~masks.get(c, pd.Series(False, index=df.index))).sum())
        base = baseline.get(c, "") if baseline else ""
        try:
            if base != "" and pd.notna(base):
                base = round(float(base), 2)
        except Exception:
            pass
        cnt = int(data.count())
        rows.append({
            "列名": c,
            "基準値": base,
            "データ数": cnt,
            "平均値": round(float(data.mean()), 2) if cnt else np.nan,
            "標準偏差": round(float(data.std(ddof=1)), 2) if cnt else np.nan,
            "最小値": float(data.min()) if cnt else np.nan,
            "最大値": float(data.max()) if cnt else np.nan,
            "欠損数": miss,
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
        mu, sd = float(s.mean()), float(s.std(ddof=1))
        if sd == 0:
            out.append({"列名": c, "平均値": round(mu, 2), "標準偏差": 0, "Cp": "計算不可", "Cpk": "計算不可"})
            continue
        cp = (usl - lsl) / (6 * sd)
        cpk = min((usl - mu) / (3 * sd), (mu - lsl) / (3 * sd))
        out.append({"列名": c, "平均値": round(mu, 2), "標準偏差": round(sd, 2), "Cp": round(cp, 3), "Cpk": round(cpk, 3)})
    return pd.DataFrame(out)

# Plotly（インタラクティブ）ヒスト
def hist_with_gauss(df: pd.DataFrame, col: str, usl: float | None, lsl: float | None):
    data = df[col].dropna()
    if data.empty:
        return None
    mu, sd = float(data.mean()), float(data.std(ddof=1))
    hist_values, bins = np.histogram(data, bins=30)
    centers = (bins[:-1] + bins[1:]) / 2
    widths = np.diff(bins)

    fig = go.Figure()
    fig.add_bar(x=centers, y=hist_values, width=widths, name="頻度", opacity=0.7)

    if sd > 0:
        xs = np.linspace(data.min(), data.max(), 200)
        pdf = stats.norm.pdf(xs, mu, sd)
        fig.add_scatter(x=xs, y=pdf, name="ガウス分布", mode="lines")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="確率密度"))

    title = f"{col} のヒストグラム" + ("" if sd > 0 else "（全データ同一）")
    fig.update_layout(title=title, xaxis_title=col, yaxis_title="頻度", hovermode="x unified", legend=dict(x=0.7, y=1))

    if usl is not None:
        fig.add_vline(x=usl, line_dash="dash", line_color="red", line_width=2, annotation_text="USL", annotation_position="top")
    if lsl is not None:
        fig.add_vline(x=lsl, line_dash="dash", line_color="orange", line_width=2, annotation_text="LSL", annotation_position="top")
    return fig

# CSV出力（文字化け対策）
def csv_bytes(df: pd.DataFrame, encoding: str = "cp932") -> bytes:
    """Excel向け(Windows)の既定はcp932。UTF-8(BOM)も選べる。"""
    csv_text = df.to_csv(index=False)
    if encoding.lower() in ("cp932", "shift_jis", "sjis"):
        return csv_text.encode("cp932", errors="replace")
    elif encoding.lower() == "utf-8-sig":
        return csv_text.encode("utf-8-sig")
    else:
        return csv_text.encode(encoding, errors="replace")

# ---------------- Matplotlib 図（PDF用） ----------------
def make_scatter_cumulative_fig(df: pd.DataFrame, x: str, color_col: str | None = None,
                                fontname: str | None = None, flip: bool = False):
    """Y=1..N（累積件数）散布図の matplotlib Figure を返す（flip=True で軸反転）"""
    df2 = df[[x]].dropna().copy()
    if pd.api.types.is_numeric_dtype(df2[x]) or pd.api.types.is_datetime64_any_dtype(df2[x]):
        df2 = df2.sort_values(by=x, kind="mergesort")
    df2["データ数"] = np.arange(1, len(df2) + 1)
    if color_col and color_col in df.columns:
        df2[color_col] = df.loc[df2.index, color_col]

    fig, ax = plt.subplots(figsize=(6, 3))
    if color_col and color_col in df2.columns:
        for name, g in df2.groupby(color_col):
            if flip: ax.scatter(g["データ数"], g[x], s=10, label=str(name))
            else:    ax.scatter(g[x], g["データ数"], s=10, label=str(name))
        ax.legend(fontsize=8, loc="best")
    else:
        if flip: ax.scatter(df2["データ数"], df2[x], s=10)
        else:    ax.scatter(df2[x], df2["データ数"], s=10)

    if fontname:
        if flip:
            ax.set_title(f"{x} vs 累積データ数", fontname=fontname)
            ax.set_xlabel("データ数", fontname=fontname)
            ax.set_ylabel(x, fontname=fontname)
        else:
            ax.set_title(f"累積データ数 vs {x}", fontname=fontname)
            ax.set_xlabel(x, fontname=fontname)
            ax.set_ylabel("データ数", fontname=fontname)
    else:
        if flip:
            ax.set_title(f"{x} vs 累積データ数"); ax.set_xlabel("データ数"); ax.set_ylabel(x)
        else:
            ax.set_title(f"累積データ数 vs {x}"); ax.set_xlabel(x); ax.set_ylabel("データ数")
    fig.tight_layout()
    return fig

def make_scatter_counts_fig(df: pd.DataFrame, x: str, fontname: str | None = None, flip: bool = False):
    """Y=各値の件数（頻度）散布図の matplotlib Figure を返す（flip=True で軸反転）"""
    df2 = df[[x]].dropna().copy()
    cnt = df2.groupby(x, dropna=False).size().reset_index(name="データ数")

    fig, ax = plt.subplots(figsize=(6, 3))
    if flip: ax.scatter(cnt["データ数"], cnt[x], s=12)
    else:    ax.scatter(cnt[x], cnt["データ数"], s=12)

    if fontname:
        if flip:
            ax.set_title(f"{x} vs 件数", fontname=fontname); ax.set_xlabel("データ数", fontname=fontname); ax.set_ylabel(x, fontname=fontname)
        else:
            ax.set_title(f"{x} ごとの件数", fontname=fontname); ax.set_xlabel(x, fontname=fontname); ax.set_ylabel("データ数", fontname=fontname)
    else:
        if flip: ax.set_title(f"{x} vs 件数"); ax.set_xlabel("データ数"); ax.set_ylabel(x)
        else:    ax.set_title(f"{x} ごとの件数"); ax.set_xlabel(x); ax.set_ylabel("データ数")
    fig.tight_layout(); return fig

def make_running_total_fig(df: pd.DataFrame, value_col: str, x_col: str | None = None,
                           fontname: str | None = None, flip: bool = False):
    """小計（ランニングサム）を描画する Matplotlib 図
       value_col: 小計対象の数値列
       x_col    : 時系列のX軸（未指定なら行番号 1..N）
       flip     : 軸反転（X↔Y）
    """
    # 行順（ファイルの並び＝時系列）を維持して抽出
    if x_col:
        mask = df[value_col].notna() & df[x_col].notna()
        df2 = df.loc[mask, [value_col, x_col]].copy()
    else:
        mask = df[value_col].notna()
        df2 = df.loc[mask, [value_col]].copy()
        df2["__idx__"] = np.arange(1, len(df2) + 1)

    df2["小計"] = pd.to_numeric(df2[value_col], errors="coerce").fillna(0).cumsum()
    X = df2[x_col] if x_col else df2["__idx__"]
    Y = df2["小計"]

    fig, ax = plt.subplots(figsize=(6, 3))
    if flip:
        ax.plot(Y, X, marker="o", linewidth=1)
    else:
        ax.plot(X, Y, marker="o", linewidth=1)

    # ラベル
    xlab = x_col if x_col else "行番号"
    if fontname:
        if flip:
            ax.set_title(f"{value_col} の小計（ランニングサム）", fontname=fontname)
            ax.set_xlabel("小計", fontname=fontname); ax.set_ylabel(xlab, fontname=fontname)
        else:
            ax.set_title(f"{value_col} の小計（ランニングサム）", fontname=fontname)
            ax.set_xlabel(xlab, fontname=fontname); ax.set_ylabel("小計", fontname=fontname)
    else:
        if flip:
            ax.set_title(f"{value_col} の小計（ランニングサム）"); ax.set_xlabel("小計"); ax.set_ylabel(xlab)
        else:
            ax.set_title(f"{value_col} の小計（ランニングサム）"); ax.set_xlabel(xlab); ax.set_ylabel("小計")
    fig.tight_layout(); return fig

# ---------------- PDF 生成（日本語対応・全件出力・各種図含む） ----------------
def _jp_paragraph_styles():
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        'TitleJP', parent=styles['Heading1'],
        fontName=RL_FONT_BOLD or styles['Heading1'].fontName,
        fontSize=20, textColor=colors.HexColor('#1f77b4'),
        alignment=1, spaceAfter=18
    )
    h2 = ParagraphStyle(
        'H2JP', parent=styles['Heading2'],
        fontName=RL_FONT_BOLD or styles['Heading2'].fontName,
        fontSize=14
    )
    normal = ParagraphStyle(
        'NormalJP', parent=styles['Normal'],
        fontName=RL_FONT_REG or styles['Normal'].fontName,
        fontSize=10
    )
    return title, h2, normal

def _jp_table_style(header_bold=True):
    base = [
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]
    if RL_FONT_REG:
        base.append(('FONTNAME', (0, 0), (-1, -1), RL_FONT_REG))
    if header_bold and RL_FONT_BOLD:
        base.append(('FONTNAME', (0, 0), (-1, 0), RL_FONT_BOLD))
    base += [
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ]
    return TableStyle(base)

def generate_pdf(
    df_clean: pd.DataFrame,
    stats_df: pd.DataFrame | None,
    sheet_name: str,
    masks: dict,
    pdf_opts: dict | None = None
) -> bytes | None:
    """pdf_opts:
        {
          'include_hist': True/False,
          'include_scatter_cum': True/False,
          'include_scatter_cnt': True/False,
          'scatter_x': str | None,
          'scatter_color': str | None,
          'scatter_flip': True/False,
          'include_running_total': True/False,
          'rt_value_col': str | None,
          'rt_x_col': str | None,
          'rt_flip': True/False
        }
    """
    pdf_opts = pdf_opts or {}
    include_hist = pdf_opts.get('include_hist', True)
    include_scatter_cum = pdf_opts.get('include_scatter_cum', False)
    include_scatter_cnt = pdf_opts.get('include_scatter_cnt', False)
    scatter_x = pdf_opts.get('scatter_x', None)
    scatter_color = pdf_opts.get('scatter_color', None)
    scatter_flip = pdf_opts.get('scatter_flip', False)

    include_running_total = pdf_opts.get('include_running_total', False)
    rt_value_col = pdf_opts.get('rt_value_col', None)
    rt_x_col = pdf_opts.get('rt_x_col', None)
    rt_flip = pdf_opts.get('rt_flip', False)

    try:
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = tmp_pdf.name
        tmp_pdf.close()
        tmp_imgs = []

        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []

        title_style, h2_style, normal_style = _jp_paragraph_styles()

        # タイトル/日付
        story += [
            Paragraph(f"Excel Data Analysis Report<br/>{sheet_name}", title_style),
            Paragraph(datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S"), normal_style),
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
        t_over = Table(overview, colWidths=[3 * inch, 2 * inch])
        t_over.setStyle(_jp_table_style())
        story += [Paragraph("Data Overview", h2_style), t_over, Spacer(1, 0.2 * inch)]

        # 基本統計量（全行）
        if stats_df is not None and not stats_df.empty:
            data = [stats_df.columns.tolist()] + [[str(v) for v in r] for r in stats_df.to_numpy()]
            t_stats = Table(data, repeatRows=1)
            t_stats.setStyle(_jp_table_style())
            story += [Paragraph("Basic Statistics", h2_style), t_stats, Spacer(1, 0.2 * inch)]

        # ヒスト（全数値列）
        if include_hist:
            ncols = numeric_cols(df_clean)
            if ncols:
                story += [PageBreak(), Paragraph("Distribution Charts", h2_style), Spacer(1, 0.1 * inch)]
                for i, c in enumerate(ncols):
                    fig, ax = plt.subplots(figsize=(6, 3))
                    df_clean[c].dropna().hist(bins=30, ax=ax, edgecolor="black")
                    if MPL_FONT_NAME:
                        ax.set_title(f"{c} Distribution", fontname=MPL_FONT_NAME)
                        ax.set_xlabel(c, fontname=MPL_FONT_NAME)
                        ax.set_ylabel("Frequency", fontname=MPL_FONT_NAME)
                    else:
                        ax.set_title(f"{c} Distribution"); ax.set_xlabel(c); ax.set_ylabel("Frequency")

                    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    fig.savefig(tmp_img.name, bbox_inches="tight", dpi=110)
                    plt.close(fig)
                    tmp_imgs.append(tmp_img.name)

                    story += [Image(tmp_img.name, width=5 * inch, height=2.5 * inch), Spacer(1, 0.2 * inch)]
                    if (i + 1) % 2 == 0 and i < len(ncols) - 1:
                        story.append(PageBreak())

        # 散布図（Y=データ数 固定、軸反転対応）
        if (include_scatter_cum or include_scatter_cnt) and scatter_x:
            story += [PageBreak(), Paragraph("Count-based Scatter Charts", h2_style), Spacer(1, 0.1 * inch)]

            if include_scatter_cum:
                fig = make_scatter_cumulative_fig(
                    df_clean, scatter_x, color_col=scatter_color, fontname=MPL_FONT_NAME, flip=scatter_flip
                )
                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmp_img.name, bbox_inches="tight", dpi=110)
                plt.close(fig)
                tmp_imgs.append(tmp_img.name)
                story += [Image(tmp_img.name, width=5 * inch, height=2.5 * inch), Spacer(1, 0.2 * inch)]

            if include_scatter_cnt:
                fig = make_scatter_counts_fig(
                    df_clean, scatter_x, fontname=MPL_FONT_NAME, flip=scatter_flip
                )
                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmp_img.name, bbox_inches="tight", dpi=110)
                plt.close(fig)
                tmp_imgs.append(tmp_img.name)
                story += [Image(tmp_img.name, width=5 * inch, height=2.5 * inch), Spacer(1, 0.2 * inch)]

        # 小計（ランニングサム） 時系列推移
        if include_running_total and rt_value_col:
            story += [PageBreak(), Paragraph("Running Total (小計) Chart", h2_style), Spacer(1, 0.1 * inch)]
            fig = make_running_total_fig(
                df_clean, value_col=rt_value_col, x_col=rt_x_col, fontname=MPL_FONT_NAME, flip=rt_flip
            )
            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp_img.name, bbox_inches="tight", dpi=110)
            plt.close(fig)
            tmp_imgs.append(tmp_img.name)
            story += [Image(tmp_img.name, width=5 * inch, height=2.5 * inch), Spacer(1, 0.2 * inch)]

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

# ---------------- メインアプリ ----------------
def main():
    st.title("📊 C3slim データ分析アプリ")
    st.markdown("---")

    # サイドバー：アップロード & CSV文字コード
    st.sidebar.header("📁 ファイルアップロード")
    up = st.sidebar.file_uploader("Excelファイルを選択 (.xlsx/.xls)", type=["xlsx", "xls"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧾 CSVの文字コード")
    st.sidebar.radio(
        "Excelで文字化けする場合は『cp932』を選択してください。",
        ["cp932 (Shift_JIS/Excel向け)", "utf-8-sig (UTF-8/BOM)"],
        index=0, horizontal=False, key="csv_enc_label"
    )
    st.session_state["csv_enc"] = "cp932" if st.session_state.csv_enc_label.startswith("cp932") else "utf-8-sig"

    # セッション初期化
    if "all_sheets" not in st.session_state: st.session_state.all_sheets = {}
    if "baseline" not in st.session_state: st.session_state.baseline = {}
    if "sheet" not in st.session_state: st.session_state.sheet = None

    # ファイル読み込み
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

    # ---------------- タブ ----------------
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
            st.download_button("📥 統計量CSV", data=csv_bytes(stats_df, st.session_state["csv_enc"]),
                               file_name="basic_statistics.csv", mime="text/csv")
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
                    st.download_button("📥 Cp/Cpk CSV", data=csv_bytes(cpk_df, st.session_state["csv_enc"]),
                                       file_name="process_capability.csv", mime="text/csv")
        else:
            st.warning("数値列が見つかりません。")

    with tab3:
        st.subheader("📊 ヒストグラム")
        ncols = numeric_cols(df_clean)
        if not ncols:
            st.warning("数値列が見つかりません。")
        else:
            col = st.selectbox("列を選択", ncols)
            base_val = base.get(col, "")
            if base_val != "" and pd.notna(base_val):
                try:
                    st.info(f"💡 基準値: {float(base_val):.3f}（USL/LSLへ転記可）")
                except Exception:
                    st.info(f"💡 基準値: {base_val}")

            c1, c2, c3 = st.columns([2, 2, 1])
            with c1: usl_text = st.text_input(f"{col} の USL", value="", key=f"usl_{col}")
            with c2: lsl_text = st.text_input(f"{col} の LSL", value="", key=f"lsl_{col}")
            with c3:
                if base_val != "" and pd.notna(base_val):
                    st.write("")
                    if st.button("📋 基準値を転記", key=f"copy_{col}"):
                        try:
                            b = str(float(base_val)); st.session_state[f"usl_{col}"] = b; st.session_state[f"lsl_{col}"] = b; st.rerun()
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

            s = df_clean[col].dropna()
            if not s.empty:
                st.markdown("#### 📊 統計サマリー")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("データ数", len(s)); c2.metric("平均値", f"{s.mean():.3f}")
                c3.metric("標準偏差", f"{s.std(ddof=1):.3f}")
                c4.metric("最小値", f"{s.min():.3f}"); c5.metric("最大値", f"{s.max():.3f}")

                if usl_v is not None and lsl_v is not None and usl_v > lsl_v:
                    sd = float(s.std(ddof=1))
                    if sd > 0:
                        cp = (usl_v - lsl_v) / (6 * sd)
                        cpk = min((usl_v - float(s.mean())) / (3 * sd), (float(s.mean()) - lsl_v) / (3 * sd))
                        st.markdown("#### 🎯 工程能力指数")
                        d1, d2 = st.columns(2)
                        d1.metric("Cp", f"{cp:.3f}"); d2.metric("Cpk", f"{cpk:.3f}")
                        if cpk >= 1.67: st.success("✅ 優良")
                        elif cpk >= 1.33: st.info("👍 良好")
                        elif cpk >= 1.00: st.warning("⚠️ 許容")
                        else: st.error("❌ 改善が必要")
                    else:
                        st.warning("標準偏差が0のため計算できません。")

    with tab4:
        st.subheader("🔵 散布図 / 時系列推移")
        ncols = numeric_cols(df_clean)
        all_cols = df_clean.columns.tolist()

        if len(ncols) < 1:
            st.warning("少なくとも1つの数値列が必要です。")
        else:
            # Y軸モード選択 & 軸反転トグル
            ctop1, ctop2 = st.columns([3, 1])
            with ctop1:
                mode = st.radio(
                    "表示モード",
                    ["通常（列を選択）", "累積データ数（1..N）", "各値の件数（Xごとの件数）", "小計（ランニングサム）"],
                    index=0, horizontal=True
                )
            with ctop2:
                flip_axes = st.checkbox("XとYを反転（X↔Y）", value=False, help="軸を入れ替えて表示します。")

            color_col = st.selectbox("色分け列（任意）", ["なし"] + all_cols, key="scatter_color_fixed")

            # -------- 通常 --------
            if mode == "通常（列を選択）":
                x = st.selectbox("X軸（数値列）", ncols, key="scatter_x_fixed")
                y_candidates = [c for c in ncols if c != x] if x in ncols else ncols
                y = st.selectbox("Y軸（数値列）", y_candidates, key="scatter_y_fixed")

                if st.button("📈 作成", type="primary", key="scatter_btn_norm"):
                    color_param = None if color_col == "なし" else color_col
                    plot_x, plot_y = (y, x) if flip_axes else (x, y)
                    fig = px.scatter(df_clean, x=plot_x, y=plot_y, color=color_param, title=f"{plot_y} vs {plot_x}")
                    fig.update_layout(title_font_size=16, xaxis_title=plot_x, yaxis_title=plot_y)
                    st.plotly_chart(fig, use_container_width=True)

                    # 相関係数
                    try:
                        corr = df_clean[[x, y]].corr().iloc[0, 1]
                        st.metric("相関係数", f"{corr:.3f}")
                        st.info("✅ 強い相関" if abs(corr) >= 0.7 else ("👍 中程度の相関" if abs(corr) >= 0.4 else "⚠️ 弱い相関"))
                    except Exception:
                        pass

            # -------- 累積データ数 --------
            elif mode == "累積データ数（1..N）":
                x = st.selectbox("X軸（任意の列）", all_cols, key="scatter_x_fixed_any")
                st.caption("Xを並べ、Y=1..N（累積件数）として描画します。")
                if st.button("📈 作成", type="primary", key="scatter_btn_cum"):
                    df2 = df_clean[[x]].dropna().copy()
                    if pd.api.types.is_numeric_dtype(df2[x]) or pd.api.types.is_datetime64_any_dtype(df2[x]):
                        df2 = df2.sort_values(by=x, kind="mergesort")
                    df2["データ数"] = np.arange(1, len(df2) + 1)
                    color_param = None if color_col == "なし" else color_col
                    if color_param and color_param in df_clean.columns:
                        df2[color_param] = df_clean.loc[df2.index, color_param]

                    if flip_axes:
                        fig = px.scatter(df2, x="データ数", y=x, color=(None if color_col == "なし" else color_col),
                                         title=f"{x} vs 累積データ数")
                        fig.update_layout(xaxis_title="データ数", yaxis_title=x)
                    else:
                        fig = px.scatter(df2, x=x, y="データ数", color=(None if color_col == "なし" else color_col),
                                         title=f"累積データ数 vs {x}")
                        fig.update_layout(xaxis_title=x, yaxis_title="データ数")
                    fig.update_layout(title_font_size=16)
                    st.plotly_chart(fig, use_container_width=True)

            # -------- 各値の件数 --------
            elif mode == "各値の件数（Xごとの件数）":
                x = st.selectbox("X軸（任意の列）", all_cols, key="scatter_x_counts")
                st.caption("X の各値（カテゴリ/数値）ごとの出現回数を Y=データ数 として描画します。")
                if st.button("📈 作成", type="primary", key="scatter_btn_freq"):
                    df2 = df_clean[[x]].dropna()
                    cnt = df2.groupby(x, dropna=False).size().reset_index(name="データ数")
                    if flip_axes:
                        fig = px.scatter(cnt, x="データ数", y=x, title=f"{x} vs 件数")
                        fig.update_layout(xaxis_title="データ数", yaxis_title=x)
                    else:
                        fig = px.scatter(cnt, x=x, y="データ数", title=f"{x} ごとの件数")
                        fig.update_layout(xaxis_title=x, yaxis_title="データ数")
                    fig.update_layout(title_font_size=16)
                    st.plotly_chart(fig, use_container_width=True)

            # -------- 小計（ランニングサム） --------
            else:
                y_val = st.selectbox("小計対象の数値列（Y）", ncols, key="rt_value_col")
                x_ts = st.selectbox("時系列X（任意/未選択=行番号）", ["（未選択）"] + all_cols, key="rt_x_col")
                x_use = None if x_ts == "（未選択）" else x_ts
                st.caption("データの並び順（上から）を時系列として、小計（累積和）を描画します。")

                if st.button("📈 作成", type="primary", key="scatter_btn_running"):
                    # マスク＆整形（並び順はそのまま）
                    if x_use:
                        mask = df_clean[y_val].notna() & df_clean[x_use].notna()
                        df2 = df_clean.loc[mask, [y_val, x_use]].copy()
                    else:
                        mask = df_clean[y_val].notna()
                        df2 = df_clean.loc[mask, [y_val]].copy()
                        df2["行番号"] = np.arange(1, len(df2) + 1)

                    df2["小計"] = pd.to_numeric(df2[y_val], errors="coerce").fillna(0).cumsum()
                    plot_x = (x_use if x_use else "行番号")
                    plot_y = "小計"

                    if flip_axes:
                        fig = px.line(df2, x=plot_y, y=plot_x, title=f"{y_val} の小計（ランニングサム）")
                        fig.update_layout(xaxis_title="小計", yaxis_title=plot_x)
                    else:
                        fig = px.line(df2, x=plot_x, y=plot_y, title=f"{y_val} の小計（ランニングサム）")
                        fig.update_layout(xaxis_title=plot_x, yaxis_title="小計")
                    fig.update_traces(mode="lines+markers")
                    fig.update_layout(title_font_size=16)
                    st.plotly_chart(fig, use_container_width=True)

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
                flip_axes_custom = st.checkbox("XとYを反転（X↔Y）", value=False, key="flip_custom")
            else:
                x = st.selectbox("X軸", all_cols, key="gx2")
                y = st.selectbox("Y軸", ncols, key="gy2")
                flip_axes_custom = False
            color_col = st.selectbox("色分け列（任意）", ["なし"] + all_cols, key="gcolor")
            if st.button("📈 生成", type="primary"):
                color_param = None if color_col == "なし" else color_col
                px_x, px_y = (y, x) if (gtype == "散布図" and flip_axes_custom) else (x, y)
                if gtype == "散布図":
                    fig = px.scatter(df_clean, x=px_x, y=px_y, color=color_param, title=f"{px_y} vs {px_x}")
                elif gtype == "折れ線":
                    fig = px.line(df_clean, x=x, y=y, color=color_param, title=f"{y} の推移")
                elif gtype == "棒":
                    fig = px.bar(df_clean, x=x, y=y, color=color_param, title=f"{y} の棒グラフ")
                else:
                    fig = px.area(df_clean, x=x, y=y, color=color_param, title=f"{y} の面グラフ")
                fig.update_layout(title_font_size=16, xaxis_title=px_x, yaxis_title=px_y)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    xv = df_clean[px_x]
                    xv_mean = f"{xv.mean():.2f}" if pd.api.types.is_numeric_dtype(xv) else "N/A"
                    st.metric(f"{px_x} 平均", xv_mean)
                with c2:
                    st.metric(f"{px_y} 平均", f"{df_clean[px_y].mean():.2f}" if pd.api.types.is_numeric_dtype(df_clean[px_y]) else "N/A")

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
                st.download_button("📥 フィルタ後CSV", data=csv_bytes(filt, st.session_state["csv_enc"]),
                                   file_name="filtered_data.csv", mime="text/csv")

    # ---------------- エクスポート ----------------
    st.markdown("---")
    st.subheader("💾 エクスポート")

    # PDF 出力オプション
    st.markdown("#### 📝 PDF オプション")
    pdf_include_hist = st.checkbox("数値列のヒストグラムをすべて含める", value=True)
    pdf_include_scatter_cum = st.checkbox("散布図（累積データ数 1..N）を含める", value=False)
    pdf_include_scatter_cnt = st.checkbox("散布図（各値の件数）を含める", value=False)
    pdf_scatter_flip = st.checkbox("（PDF）散布図のXとYを反転（X↔Y）", value=False)

    # 小計（ランニングサム）PDF設定
    st.markdown("##### 小計（ランニングサム）をPDFに含める")
    pdf_include_rt = st.checkbox("小計（ランニングサム）チャートを含める", value=False)
    rt_value_default = numeric_cols(df_clean)[0] if numeric_cols(df_clean) else None
    col_rt1, col_rt2, col_rt3 = st.columns([2, 2, 1])
    with col_rt1:
        rt_value_col_pdf = st.selectbox("小計対象の数値列（Y）", numeric_cols(df_clean), index=0 if rt_value_default else 0)
    with col_rt2:
        rt_x_col_pdf = st.selectbox("時系列X（任意/未選択=行番号）", ["（未選択）"] + df_clean.columns.tolist(), index=0)
        if rt_x_col_pdf == "（未選択）":
            rt_x_col_pdf = None
    with col_rt3:
        rt_flip_pdf = st.checkbox("（PDF）小計のXとY反転", value=False)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("📥 元データCSV", data=csv_bytes(df_raw, st.session_state["csv_enc"]),
                           file_name="original_data.csv", mime="text/csv")
    with c2:
        st.download_button("📥 クリーニング後CSV", data=csv_bytes(df_clean, st.session_state["csv_enc"]),
                           file_name="cleaned_data.csv", mime="text/csv")
    with c3:
        if st.button("📄 PDFレポート生成", type="primary"):
            with st.spinner("PDFを生成中..."):
                stats_df = basic_stats(df_clean, masks, base)
                pdf = generate_pdf(
                    df_clean, stats_df, st.session_state.sheet or "データ", masks,
                    pdf_opts={
                        'include_hist': pdf_include_hist,
                        'include_scatter_cum': pdf_include_scatter_cum,
                        'include_scatter_cnt': pdf_include_scatter_cnt,
                        'scatter_x': st.session_state.get("scatter_x_fixed_any") or st.session_state.get("scatter_x_counts"),
                        'scatter_color': None if st.session_state.get("scatter_color_fixed") == "なし" else st.session_state.get("scatter_color_fixed"),
                        'scatter_flip': pdf_scatter_flip,
                        'include_running_total': pdf_include_rt,
                        'rt_value_col': rt_value_col_pdf,
                        'rt_x_col': rt_x_col_pdf,
                        'rt_flip': rt_flip_pdf
                    }
                )
                if pdf:
                    st.download_button("📥 PDFレポート", pdf,
                                       f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                       "application/pdf")

if __name__ == "__main__":
    main()
