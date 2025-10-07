import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
import io
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import tempfile
import os

# 警告を非表示
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="Excel データ分析アプリ",
    page_icon="📊",
    layout="wide"
)

# 日本語フォント設定（matplotlib用）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# セッション状態の初期化
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'all_sheets' not in st.session_state:
    st.session_state.all_sheets = {}
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None
if 'non_missing_exceptions' not in st.session_state:
    st.session_state.non_missing_exceptions = {}
if 'baseline_values' not in st.session_state:
    st.session_state.baseline_values = {}

def clean_data(df):
    """データクリーニング関数"""
    try:
        # データのコピーを作成
        df_clean = df.copy()
        
        # 空行や不要な行を除去
        df_clean = df_clean.dropna(how='all')
        
        # インデックスをリセット
        df_clean = df_clean.reset_index(drop=True)
        
        # 文字列"None"や空白を欠損として扱わないためのマスク作成
        non_missing_masks = {}
        
        # 数値列を自動検出して変換
        for col in df_clean.columns:
            # 数値変換前に、文字列"None"や空白をマスク
            if df_clean[col].dtype == 'object':
                # 文字列として扱い、"None"（大文字小文字区別なし）または空白（前後空白削除後）をマスク
                str_series = df_clean[col].astype(str)
                is_string_none = str_series.str.lower() == 'none'
                is_whitespace = str_series.str.strip() == ''
                # これらは欠損として扱わない（有効な値として扱う）
                non_missing_masks[col] = is_string_none | is_whitespace
            else:
                # 数値型の列はマスクなし
                non_missing_masks[col] = pd.Series([False] * len(df_clean), index=df_clean.index)
            
            # 数値に変換可能な列を検出
            try:
                # まず文字列型の場合は数値変換を試行
                if df_clean[col].dtype == 'object':
                    # カンマを除去して数値変換を試行
                    temp_series = pd.to_numeric(df_clean[col].astype(str).str.replace(',', ''), errors='coerce')
                    # 変換成功率が50%以上の場合は数値列として扱う
                    if temp_series.notna().sum() / len(temp_series) > 0.5:
                        df_clean[col] = temp_series
                else:
                    # 既に数値型の場合はそのまま
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            except:
                pass
        
        # マスクをセッション状態に保存
        st.session_state.non_missing_exceptions = non_missing_masks
        
        return df_clean
    except Exception as e:
        st.error(f"データクリーニング中にエラーが発生しました: {str(e)}")
        return df

def get_basic_stats(df, exception_masks=None, baseline_values=None):
    """基本統計量を計算"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None
    
    # 空データのチェック
    if len(df) == 0:
        return None
    
    stats_data = []
    for col in numeric_cols:
        # 欠損数の計算（マスクがある場合は調整）
        if exception_masks and col in exception_masks:
            # 真の欠損のみカウント（文字列"None"や空白は除外）
            missing_count = (df[col].isna() & (~exception_masks[col])).sum()
        else:
            # マスクがない場合は通常のカウント
            missing_count = df[col].isnull().sum()
        
        missing_percent = round(missing_count / len(df) * 100, 2)
        
        # 基準値を取得
        baseline_val = ""
        if baseline_values and col in baseline_values:
            baseline_val = baseline_values.get(col, "")
            # 数値に変換可能な場合は丸める
            try:
                if pd.notna(baseline_val) and baseline_val != "":
                    baseline_val = round(float(baseline_val), 2)
            except:
                pass
        
        stats_data.append({
            '列名': col,
            '基準値': baseline_val,
            'データ数': df[col].count(),
            '平均値': df[col].mean().round(2) if df[col].count() > 0 else np.nan,
            '標準偏差': df[col].std().round(2) if df[col].count() > 0 else np.nan,
            '最小値': df[col].min() if df[col].count() > 0 else np.nan,
            '最大値': df[col].max() if df[col].count() > 0 else np.nan,
            '欠損数': missing_count,
            '欠損率(%)': missing_percent
        })
    
    return pd.DataFrame(stats_data)

def create_histogram_plotly(df, column, usl=None, lsl=None):
    """Plotlyでヒストグラムとガウス分布を作成"""
    # データの準備
    data = df[column].dropna()
    
    if len(data) == 0:
        return None
    
    # 基本統計量の計算
    mean = data.mean()
    std = data.std()
    
    # ヒストグラムのビン設定
    hist_values, bin_edges = np.histogram(data, bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)
    
    # グラフ作成（2つのY軸）
    fig = go.Figure()
    
    # ヒストグラム（左Y軸：頻度）
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist_values,
        width=bin_widths,
        name='頻度',
        marker_color='lightblue',
        yaxis='y1',
        opacity=0.7
    ))
    
    # ガウス分布曲線（右Y軸：確率密度）を標準偏差が0より大きい場合のみ追加
    if std > 0:
        x_range = np.linspace(data.min(), data.max(), 200)
        pdf = stats.norm.pdf(x_range, mean, std)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=pdf,
            name='ガウス分布',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        # レイアウト設定（2軸）
        fig.update_layout(
            title=dict(text=f'{column} のヒストグラムとガウス分布', font=dict(size=16)),
            xaxis=dict(
                title=dict(text=column, font=dict(size=14))
            ),
            yaxis=dict(
                title=dict(text='頻度', font=dict(size=14, color='blue')),
                tickfont=dict(color='blue')
            ),
            yaxis2=dict(
                title=dict(text='確率密度', font=dict(size=14, color='red')),
                tickfont=dict(color='red'),
                overlaying='y',
                side='right'
            ),
            legend=dict(
                x=0.7,
                y=1.0,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            hovermode='x unified'
        )
    else:
        # 標準偏差が0の場合はヒストグラムのみ表示
        fig.update_layout(
            title=dict(text=f'{column} のヒストグラム（全データが同じ値）', font=dict(size=16)),
            xaxis=dict(
                title=dict(text=column, font=dict(size=14))
            ),
            yaxis=dict(
                title=dict(text='頻度', font=dict(size=14, color='blue')),
                tickfont=dict(color='blue')
            ),
            hovermode='x unified'
        )
    
    # USL/LSLの縦線を追加
    if usl is not None:
        fig.add_vline(
            x=usl, 
            line_dash="dash", 
            line_color="red", 
            line_width=2,
            annotation_text="USL",
            annotation_position="top"
        )
    
    if lsl is not None:
        fig.add_vline(
            x=lsl, 
            line_dash="dash", 
            line_color="orange", 
            line_width=2,
            annotation_text="LSL",
            annotation_position="top"
        )
    
    return fig

def generate_pdf_report(df_clean, stats_df, sheet_name="データ", exception_masks=None):
    """PDFレポートを生成"""
    try:
        # 一時ファイルを作成
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf_filename = temp_pdf.name
        temp_pdf.close()
        
        # 画像保存用リスト
        temp_images = []
        
        # PDFドキュメントを作成
        doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # タイトル
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1  # Center
        )
        title = Paragraph(f"Excel Data Analysis Report<br/>{sheet_name}", title_style)
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # 生成日時
        date_style = ParagraphStyle('DateStyle', parent=styles['Normal'], fontSize=10, alignment=1)
        date_text = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style)
        story.append(date_text)
        story.append(Spacer(1, 0.5*inch))
        
        # データ概要
        overview_style = ParagraphStyle('OverviewStyle', parent=styles['Heading2'], fontSize=14)
        overview_title = Paragraph("Data Overview", overview_style)
        story.append(overview_title)
        story.append(Spacer(1, 0.2*inch))
        
        # 調整後の欠損数を計算
        total_missing = 0
        if exception_masks:
            for col in df_clean.columns:
                if col in exception_masks:
                    total_missing += (df_clean[col].isna() & (~exception_masks[col])).sum()
                else:
                    total_missing += df_clean[col].isnull().sum()
        else:
            total_missing = df_clean.isnull().sum().sum()
        
        overview_data = [
            ['Metric', 'Value'],
            ['Total Rows', str(len(df_clean))],
            ['Total Columns', str(len(df_clean.columns))],
            ['Numeric Columns', str(len(df_clean.select_dtypes(include=[np.number]).columns))],
            ['Missing Values', str(total_missing)]
        ]
        
        overview_table = Table(overview_data, colWidths=[3*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 0.5*inch))
        
        # 基本統計量
        if stats_df is not None and len(stats_df) > 0:
            stats_title = Paragraph("Basic Statistics", overview_style)
            story.append(stats_title)
            story.append(Spacer(1, 0.2*inch))
            
            # 統計テーブル作成（最初の10列まで）
            stats_data = [stats_df.columns.tolist()]
            for _, row in stats_df.head(10).iterrows():
                stats_data.append([str(val) for val in row.tolist()])
            
            stats_table = Table(stats_data, repeatRows=1)
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(stats_table)
            story.append(Spacer(1, 0.3*inch))
        
        # ヒストグラム画像を追加（数値列）
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            story.append(PageBreak())
            chart_title = Paragraph("Distribution Charts", overview_style)
            story.append(chart_title)
            story.append(Spacer(1, 0.2*inch))
            
            for i, col in enumerate(numeric_cols[:6]):  # 最初の6列まで
                # Matplotlibでヒストグラム作成
                fig, ax = plt.subplots(figsize=(6, 3))
                df_clean[col].dropna().hist(bins=30, ax=ax, edgecolor='black')
                ax.set_title(f'{col} Distribution')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                
                # 画像を一時ファイルに保存
                temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_images.append(temp_img.name)
                plt.savefig(temp_img.name, bbox_inches='tight', dpi=100)
                plt.close()
                
                # PDFに追加
                img = Image(temp_img.name, width=5*inch, height=2.5*inch)
                story.append(img)
                story.append(Spacer(1, 0.3*inch))
                
                # ページ区切り（2つごと）
                if (i + 1) % 2 == 0 and i < len(numeric_cols) - 1:
                    story.append(PageBreak())
        
        # PDFを構築
        doc.build(story)
        
        # PDFファイルを読み込んで返す
        with open(pdf_filename, 'rb') as f:
            pdf_data = f.read()
        
        # 一時ファイルを削除
        os.unlink(pdf_filename)
        for temp_img in temp_images:
            try:
                os.unlink(temp_img)
            except:
                pass
        
        return pdf_data
        
    except Exception as e:
        st.error(f"PDF生成エラー: {str(e)}")
        return None

# メインアプリケーション
def main():
    st.title("📊 Excel データ分析アプリ")
    st.markdown("---")
    
    # サイドバー
    st.sidebar.header("📁 ファイルアップロード")
    uploaded_file = st.sidebar.file_uploader(
        "Excelファイルを選択してください",
        type=['xlsx', 'xls'],
        help="Excel形式のファイル(.xlsx, .xls)をアップロードできます"
    )
    
    if uploaded_file is not None:
        try:
            # Excelファイルの読み込み（全シート）
            with st.spinner('ファイルを読み込み中...'):
                # 全シート名を取得
                excel_file = pd.ExcelFile(uploaded_file)
                sheet_names = excel_file.sheet_names
                
                # 全シートを読み込み
                all_sheets = {}
                all_baseline_values = {}
                
                for sheet_name in sheet_names:
                    # 1行目を列名として読み込み（excel_file.parse()を使用してファイルポインターの問題を回避）
                    df_full = excel_file.parse(sheet_name=sheet_name, header=0)
                    
                    # 基準値候補を取得（3行目、インデックス1）
                    baseline_dict = {}
                    if len(df_full) >= 2:
                        baseline_row = df_full.iloc[1]
                        # 数値に変換可能な値が含まれているかチェック
                        numeric_count = 0
                        total_count = 0
                        for val in baseline_row:
                            if pd.notna(val):
                                total_count += 1
                                try:
                                    float(val)
                                    numeric_count += 1
                                except:
                                    pass
                        
                        # 50%以上が数値の場合のみ基準値として保存
                        if total_count > 0 and numeric_count / total_count >= 0.5:
                            baseline_dict = baseline_row.to_dict()
                    all_baseline_values[sheet_name] = baseline_dict
                    
                    # データ開始行を探す（3行目以降で最初の非空白行）
                    data_start_idx = 2  # デフォルトは3行目以降
                    if len(df_full) > 2:
                        for idx in range(2, len(df_full)):
                            # 全列がNaNでない行を見つける
                            if not df_full.iloc[idx].isna().all():
                                data_start_idx = idx
                                break
                    
                    # データ開始行以降をデータとして扱う
                    df_temp = df_full.iloc[data_start_idx:].reset_index(drop=True)
                    all_sheets[sheet_name] = df_temp
                
                st.session_state.all_sheets = all_sheets
                st.session_state.baseline_values = all_baseline_values
                
                # デフォルトで最初のシートを選択
                if st.session_state.selected_sheet is None or st.session_state.selected_sheet not in sheet_names:
                    st.session_state.selected_sheet = sheet_names[0]
                
                df = all_sheets[st.session_state.selected_sheet]
                st.session_state.df = df
                st.session_state.df_clean = clean_data(df)
            
            st.sidebar.success(f"✅ ファイルが正常に読み込まれました\n（シート数: {len(sheet_names)}）")
            
            # シート選択
            if len(sheet_names) > 1:
                st.sidebar.markdown("---")
                st.sidebar.subheader("📑 シート選択")
                selected = st.sidebar.selectbox(
                    "分析するシートを選択:",
                    sheet_names,
                    index=sheet_names.index(st.session_state.selected_sheet),
                    key="sheet_selector"
                )
                
                if selected != st.session_state.selected_sheet:
                    st.session_state.selected_sheet = selected
                    st.session_state.df = all_sheets[selected]
                    st.session_state.df_clean = clean_data(all_sheets[selected])
                    st.rerun()
                
                st.sidebar.info(f"📄 現在のシート: **{selected}**\n（{len(st.session_state.df)}行 × {len(st.session_state.df.columns)}列）")
            
        except Exception as e:
            st.sidebar.error(f"❌ ファイルの読み込みに失敗しました: {str(e)}")
            return
    
    # メインコンテンツ
    if st.session_state.df is not None:
        df = st.session_state.df
        df_clean = st.session_state.df_clean
        
        # タブでコンテンツを整理
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 データプレビュー", 
            "📈 基本統計量", 
            "📊 ヒストグラム", 
            "🎨 カスタムグラフ",
            "🔍 データフィルタリング"
        ])
        
        with tab1:
            st.header("📋 データプレビュー")
            
            # データ処理の説明
            st.info("ℹ️ Excelファイルの1行目を列名として使用し、空白行を自動的にスキップしてデータを読み込んでいます。")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔍 元データ")
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"表示: 上位10行 / 全{len(df)}行")
                
            with col2:
                st.subheader("🧹 クリーニング後")
                st.dataframe(df_clean.head(10), use_container_width=True)
                st.caption(f"表示: 上位10行 / 全{len(df_clean)}行")
            
            # データ型情報
            st.subheader("📋 データ型情報")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**元データ**")
                dtype_info = pd.DataFrame({
                    '列名': df.columns,
                    'データ型': [str(dtype) for dtype in df.dtypes],
                    '欠損値': df.isnull().sum().values
                })
                st.dataframe(dtype_info, use_container_width=True)
                
            with col2:
                st.write("**クリーニング後**")
                # 欠損値の調整カウント
                adjusted_missing = []
                exception_masks = st.session_state.non_missing_exceptions
                for col in df_clean.columns:
                    if exception_masks and col in exception_masks:
                        # 真の欠損のみカウント（文字列"None"や空白は除外）
                        missing_count = (df_clean[col].isna() & (~exception_masks[col])).sum()
                    else:
                        missing_count = df_clean[col].isnull().sum()
                    adjusted_missing.append(missing_count)
                
                dtype_info_clean = pd.DataFrame({
                    '列名': df_clean.columns,
                    'データ型': [str(dtype) for dtype in df_clean.dtypes],
                    '欠損値': adjusted_missing
                })
                st.dataframe(dtype_info_clean, use_container_width=True)
        
        with tab2:
            st.header("📈 基本統計量")
            
            # 現在のシートの基準値を取得
            current_baseline = st.session_state.baseline_values.get(st.session_state.selected_sheet, {})
            stats_df = get_basic_stats(df_clean, st.session_state.non_missing_exceptions, current_baseline)
            if stats_df is not None:
                st.dataframe(stats_df, use_container_width=True)
                
                # 統計量のダウンロード
                csv_stats = stats_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 統計量をCSVでダウンロード",
                    data=csv_stats,
                    file_name="basic_statistics.csv",
                    mime="text/csv"
                )
                
                # 工程能力指数（Cp/Cpk）セクション
                st.markdown("---")
                st.subheader("🎯 工程能力指数 (Cp, Cpk)")
                
                st.write("規格上限値（USL）と規格下限値（LSL）を入力してください：")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    usl_input = st.number_input(
                        "規格上限値 (USL)", 
                        value=0.0,
                        step=0.1,
                        format="%.2f",
                        help="Upper Specification Limit"
                    )
                with col2:
                    lsl_input = st.number_input(
                        "規格下限値 (LSL)", 
                        value=0.0,
                        step=0.1,
                        format="%.2f",
                        help="Lower Specification Limit"
                    )
                with col3:
                    st.write("")  # スペース調整
                    calculate_cpk = st.button("計算する", type="primary")
                
                # Cp/Cpk計算
                if calculate_cpk:
                    if usl_input <= lsl_input:
                        st.error("❌ 規格上限値（USL）は規格下限値（LSL）より大きい値を入力してください。")
                    else:
                        # 数値列のCp/Cpkを計算
                        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            cpk_data = []
                            for col in numeric_cols:
                                data = df_clean[col].dropna()
                                if len(data) > 1:
                                    mean = data.mean()
                                    std = data.std(ddof=1)  # サンプル標準偏差
                                    
                                    if std > 0:
                                        # Cp = (USL - LSL) / (6σ)
                                        cp = (usl_input - lsl_input) / (6 * std)
                                        
                                        # Cpk = min((USL - μ) / (3σ), (μ - LSL) / (3σ))
                                        cpu = (usl_input - mean) / (3 * std)
                                        cpl = (mean - lsl_input) / (3 * std)
                                        cpk = min(cpu, cpl)
                                        
                                        cpk_data.append({
                                            '列名': col,
                                            '平均値': round(mean, 2),
                                            '標準偏差': round(std, 2),
                                            'Cp': round(cp, 3),
                                            'Cpk': round(cpk, 3)
                                        })
                                    else:
                                        cpk_data.append({
                                            '列名': col,
                                            '平均値': round(data.mean(), 2),
                                            '標準偏差': 0,
                                            'Cp': '計算不可',
                                            'Cpk': '計算不可'
                                        })
                                else:
                                    cpk_data.append({
                                        '列名': col,
                                        '平均値': 'N/A',
                                        '標準偏差': 'N/A',
                                        'Cp': '計算不可',
                                        'Cpk': '計算不可'
                                    })
                            
                            cpk_df = pd.DataFrame(cpk_data)
                            st.dataframe(cpk_df, use_container_width=True)
                            
                            # Cp/Cpkの解釈ガイド
                            st.info("""
                            **工程能力指数の目安：**
                            - Cp, Cpk ≥ 1.67: 優良
                            - Cp, Cpk ≥ 1.33: 良好
                            - Cp, Cpk ≥ 1.00: 許容範囲
                            - Cp, Cpk < 1.00: 改善が必要
                            """)
                            
                            # ダウンロード
                            csv_cpk = cpk_df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="📥 Cp/CpkをCSVでダウンロード",
                                data=csv_cpk,
                                file_name="process_capability.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("⚠️ 数値列が見つかりませんでした。")
                else:
                    st.info("💡 USLとLSLの両方を入力するとCp/Cpkが計算されます。")
            else:
                st.warning("⚠️ 数値列が見つかりませんでした。")
        
        with tab3:
            st.header("📊 ヒストグラム")
            
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox(
                    "表示する列を選択してください:",
                    numeric_cols,
                    help="数値列のみ選択可能です"
                )
                
                if selected_col:
                    # 規格値入力セクション
                    st.subheader("🎯 規格値設定")
                    
                    # 現在のシートの基準値を取得
                    current_baseline = st.session_state.baseline_values.get(st.session_state.selected_sheet, {})
                    baseline_value = current_baseline.get(selected_col, "")
                    
                    # 基準値が存在する場合は表示
                    if baseline_value != "" and pd.notna(baseline_value):
                        try:
                            baseline_float = float(baseline_value)
                            st.info(f"💡 基準値: {baseline_float:.2f} （自動転記ボタンでUSL/LSLに設定できます）")
                        except:
                            st.info(f"💡 基準値: {baseline_value}")
                    
                    spec_col1, spec_col2, spec_col3 = st.columns([2, 2, 1])
                    with spec_col1:
                        usl_text = st.text_input(
                            f"{selected_col} の規格上限値 (USL)", 
                            value="",
                            placeholder="例: 100.0",
                            key=f"usl_{selected_col}"
                        )
                    with spec_col2:
                        lsl_text = st.text_input(
                            f"{selected_col} の規格下限値 (LSL)", 
                            value="",
                            placeholder="例: 50.0",
                            key=f"lsl_{selected_col}"
                        )
                    with spec_col3:
                        # 基準値を転記するボタン
                        if baseline_value != "" and pd.notna(baseline_value):
                            st.write("")  # スペーサー
                            if st.button("📋 基準値を転記", key=f"copy_baseline_{selected_col}"):
                                try:
                                    baseline_float = float(baseline_value)
                                    st.session_state[f"usl_{selected_col}"] = str(baseline_float)
                                    st.session_state[f"lsl_{selected_col}"] = str(baseline_float)
                                    st.rerun()
                                except:
                                    st.error("基準値を数値に変換できません")
                    
                    # 入力値をfloatに変換（空白の場合はNone）
                    usl_value = None
                    lsl_value = None
                    
                    try:
                        if usl_text.strip():
                            usl_value = float(usl_text)
                    except ValueError:
                        st.error(f"❌ USLは数値で入力してください: '{usl_text}'")
                    
                    try:
                        if lsl_text.strip():
                            lsl_value = float(lsl_text)
                    except ValueError:
                        st.error(f"❌ LSLは数値で入力してください: '{lsl_text}'")
                    
                    # 入力バリデーション
                    if usl_value is not None and lsl_value is not None and usl_value <= lsl_value:
                        st.warning("⚠️ 規格上限値（USL）は規格下限値（LSL）より大きい値を入力してください。")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        fig = create_histogram_plotly(df_clean, selected_col, usl_value, lsl_value)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("⚠️ 選択した列にデータがありません。")
                    
                    with col2:
                        st.subheader("📊 統計サマリー")
                        col_data = df_clean[selected_col].dropna()
                        if len(col_data) > 0:
                            mean_val = col_data.mean()
                            std_val = col_data.std(ddof=1)  # サンプル標準偏差
                            
                            st.metric("データ数", len(col_data))
                            st.metric("平均値", f"{mean_val:.2f}")
                            st.metric("標準偏差", f"{std_val:.2f}")
                            st.metric("最小値", f"{col_data.min():.2f}")
                            st.metric("最大値", f"{col_data.max():.2f}")
                            
                            # Cp/Cpk計算（USLとLSLが両方入力されている場合）
                            if usl_value is not None and lsl_value is not None and usl_value > lsl_value:
                                st.markdown("---")
                                st.subheader("🎯 工程能力指数")
                                
                                if std_val > 0:
                                    # Cp = (USL - LSL) / (6σ)
                                    cp = (usl_value - lsl_value) / (6 * std_val)
                                    
                                    # Cpk = min((USL - μ) / (3σ), (μ - LSL) / (3σ))
                                    cpu = (usl_value - mean_val) / (3 * std_val)
                                    cpl = (mean_val - lsl_value) / (3 * std_val)
                                    cpk = min(cpu, cpl)
                                    
                                    st.metric("Cp", f"{cp:.3f}")
                                    st.metric("Cpk", f"{cpk:.3f}")
                                    
                                    # 評価
                                    if cpk >= 1.67:
                                        st.success("✅ 優良")
                                    elif cpk >= 1.33:
                                        st.info("👍 良好")
                                    elif cpk >= 1.00:
                                        st.warning("⚠️ 許容範囲")
                                    else:
                                        st.error("❌ 改善が必要")
                                else:
                                    st.warning("標準偏差が0のため計算できません")
            else:
                st.warning("⚠️ 数値列が見つかりませんでした。")
        
        with tab4:
            st.header("🎨 カスタムグラフ")
            
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df_clean.columns.tolist()
            
            if len(numeric_cols) > 0:
                st.subheader("📊 グラフ設定")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    graph_type = st.selectbox(
                        "グラフの種類を選択:",
                        ["散布図", "折れ線グラフ", "棒グラフ", "面グラフ"]
                    )
                
                with col2:
                    if graph_type in ["散布図"]:
                        x_axis = st.selectbox("X軸:", numeric_cols, key="x_axis_custom")
                        y_axis = st.selectbox("Y軸:", [col for col in numeric_cols if col != x_axis], key="y_axis_custom")
                    else:
                        x_axis = st.selectbox("X軸:", all_cols, key="x_axis_custom2")
                        y_axis = st.selectbox("Y軸:", numeric_cols, key="y_axis_custom2")
                
                # カラー設定
                color_col = st.selectbox(
                    "色分け列（オプション）:",
                    ["なし"] + all_cols,
                    key="color_custom"
                )
                
                # グラフ生成
                if st.button("📈 グラフを生成", type="primary"):
                    try:
                        color_param = None if color_col == "なし" else color_col
                        
                        if graph_type == "散布図":
                            fig = px.scatter(
                                df_clean,
                                x=x_axis,
                                y=y_axis,
                                color=color_param,
                                title=f"{y_axis} vs {x_axis}",
                                labels={x_axis: x_axis, y_axis: y_axis}
                            )
                        elif graph_type == "折れ線グラフ":
                            fig = px.line(
                                df_clean,
                                x=x_axis,
                                y=y_axis,
                                color=color_param,
                                title=f"{y_axis} の推移",
                                labels={x_axis: x_axis, y_axis: y_axis}
                            )
                        elif graph_type == "棒グラフ":
                            fig = px.bar(
                                df_clean,
                                x=x_axis,
                                y=y_axis,
                                color=color_param,
                                title=f"{y_axis} の棒グラフ",
                                labels={x_axis: x_axis, y_axis: y_axis}
                            )
                        elif graph_type == "面グラフ":
                            fig = px.area(
                                df_clean,
                                x=x_axis,
                                y=y_axis,
                                color=color_param,
                                title=f"{y_axis} の面グラフ",
                                labels={x_axis: x_axis, y_axis: y_axis}
                            )
                        
                        fig.update_layout(
                            title_font_size=16,
                            xaxis_title_font_size=14,
                            yaxis_title_font_size=14
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 統計情報表示
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{x_axis} 平均", f"{df_clean[x_axis].mean():.2f}" if pd.api.types.is_numeric_dtype(df_clean[x_axis]) else "N/A")
                        with col2:
                            st.metric(f"{y_axis} 平均", f"{df_clean[y_axis].mean():.2f}")
                        
                    except Exception as e:
                        st.error(f"❌ グラフ生成に失敗しました: {str(e)}")
            else:
                st.warning("⚠️ 数値列が見つかりませんでした。")
        
        with tab5:
            st.header("🔍 データフィルタリング")
            
            st.subheader("📋 フィルター条件設定")
            
            # 列選択
            filter_col = st.selectbox("フィルターする列:", df_clean.columns.tolist())
            
            if pd.api.types.is_numeric_dtype(df_clean[filter_col]):
                # 数値列の場合：範囲フィルター
                col_min = float(df_clean[filter_col].min())
                col_max = float(df_clean[filter_col].max())
                
                filter_range = st.slider(
                    f"{filter_col} の範囲:",
                    min_value=col_min,
                    max_value=col_max,
                    value=(col_min, col_max),
                    key="filter_range"
                )
                
                filtered_df = df_clean[
                    (df_clean[filter_col] >= filter_range[0]) & 
                    (df_clean[filter_col] <= filter_range[1])
                ]
            else:
                # カテゴリ列の場合：値選択
                unique_values = df_clean[filter_col].unique().tolist()
                selected_values = st.multiselect(
                    f"{filter_col} の値を選択:",
                    unique_values,
                    default=unique_values,
                    key="filter_values"
                )
                
                filtered_df = df_clean[df_clean[filter_col].isin(selected_values)]
            
            # フィルター結果表示
            st.subheader("📊 フィルター結果")
            st.info(f"💡 元データ: {len(df_clean)}行 → フィルター後: {len(filtered_df)}行")
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # フィルター後データの統計
            if len(filtered_df) > 0:
                st.subheader("📈 フィルター後の基本統計量")
                # フィルター後のデータには同じマスクを使用（インデックスが保持されている）
                current_baseline = st.session_state.baseline_values.get(st.session_state.selected_sheet, {})
                filtered_stats = get_basic_stats(filtered_df, st.session_state.non_missing_exceptions, current_baseline)
                if filtered_stats is not None:
                    st.dataframe(filtered_stats, use_container_width=True)
                    
                    # ダウンロード
                    csv_filtered = filtered_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 フィルター後データをCSVでダウンロード",
                        data=csv_filtered,
                        file_name="filtered_data.csv",
                        mime="text/csv"
                    )
        
        # データのダウンロード
        st.markdown("---")
        st.subheader("💾 データのエクスポート")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # 元データのダウンロード
            csv_original = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 元データをCSVでダウンロード",
                data=csv_original,
                file_name="original_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # クリーニング後データのダウンロード
            csv_clean = df_clean.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 クリーニング後データをCSVでダウンロード",
                data=csv_clean,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        
        with col3:
            # PDFレポートのダウンロード
            if st.button("📄 PDFレポート生成", type="primary"):
                with st.spinner('PDFレポートを生成中...'):
                    current_baseline = st.session_state.baseline_values.get(st.session_state.selected_sheet, {})
                    stats_df = get_basic_stats(df_clean, st.session_state.non_missing_exceptions, current_baseline)
                    sheet_name = st.session_state.selected_sheet if st.session_state.selected_sheet else "データ"
                    pdf_data = generate_pdf_report(df_clean, stats_df, sheet_name, st.session_state.non_missing_exceptions)
                    
                    if pdf_data:
                        st.download_button(
                            label="📥 PDFレポートをダウンロード",
                            data=pdf_data,
                            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
    
    else:
        # ファイルがアップロードされていない場合
        st.info("👈 左側のサイドバーからExcelファイルをアップロードしてください。")
        
        # 使用方法の説明
        st.markdown("""
        ## 📖 使用方法
        
        1. **ファイルアップロード**: 左側のサイドバーからExcelファイル(.xlsx, .xls)を選択
        2. **データプレビュー**: アップロードしたデータの内容と構造を確認
        3. **基本統計量**: 数値列の平均、標準偏差、最小値、最大値などを表示
        4. **ヒストグラム**: 各列の分布を視覚的に確認
        5. **カスタムグラフ**: 散布図、折れ線、棒、面グラフを自由に作成
        6. **データフィルタリング**: 条件に応じてデータを絞り込み
        7. **データエクスポート**: 分析結果をCSVやPDFでダウンロード
        
        ## ✨ 機能
        
        - 📊 **インタラクティブな可視化**: Plotlyによる動的なグラフ
        - 🧹 **自動データクリーニング**: 数値型変換と欠損値処理
        - 📈 **包括的な統計分析**: 基本統計量とヒストグラム分析
        - 🎨 **カスタムグラフ作成**: 複数のグラフタイプで自由に可視化
        - 💾 **柔軟なエクスポート**: 元データと処理済みデータをCSV/PDFでダウンロード可能
        - 🌏 **日本語対応**: 完全な日本語インターフェース
        """)

if __name__ == "__main__":
    main()
