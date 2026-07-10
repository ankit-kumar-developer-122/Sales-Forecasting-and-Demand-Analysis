"""
Sales Forecasting & Demand Intelligence Dashboard
Professional Streamlit Application with Plotly Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION & CUSTOM CSS
# ============================================================
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F5F9FF 100%);
        border: 1px solid #E3F2FD;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 2px 8px rgba(13, 71, 161, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(13, 71, 161, 0.12);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #546E7A;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0D47A1;
        line-height: 1.2;
    }
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    .kpi-delta.positive { color: #2E7D32; }
    .kpi-delta.negative { color: #C62828; }
    .kpi-delta.neutral { color: #546E7A; }
    
    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0D47A1;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E3F2FD;
    }
    .section-subheader {
        font-size: 1rem;
        font-weight: 600;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Metric cards in sidebar */
    .sidebar-metric {
        background: #E3F2FD;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .sidebar-metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #546E7A;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .sidebar-metric-value {
        font-size: 1rem;
        font-weight: 700;
        color: #0D47A1;
    }
    
    /* Info boxes */
    .info-box {
        background: #E3F2FD;
        border-left: 4px solid #1976D2;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .info-box-title {
        font-weight: 600;
        color: #0D47A1;
        margin-bottom: 0.25rem;
    }
    .info-box-text {
        color: #37474F;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Strategy cards */
    .strategy-card {
        background: #FFFFFF;
        border: 1px solid #E3F2FD;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(13, 71, 161, 0.06);
    }
    .strategy-card h4 {
        color: #0D47A1;
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }
    .strategy-card p {
        color: #37474F;
        font-size: 0.85rem;
        margin: 0;
        line-height: 1.6;
    }
    .strategy-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .badge-high { background: #E8F5E9; color: #2E7D32; }
    .badge-growing { background: #E3F2FD; color: #1565C0; }
    .badge-volatile { background: #FFF3E0; color: #E65100; }
    .badge-declining { background: #FCE4EC; color: #C62828; }
    
    /* Table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: #0D47A1;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stDownloadButton > button:hover {
        background: #1565C0;
    }
    
    /* Selectbox/Slider styling */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    .stSlider > div > div > div > div {
        background: #0D47A1;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: #FFFFFF;
        border: 1px solid #E3F2FD;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #546E7A;
    }
    .stTabs [aria-selected="true"] {
        background: #0D47A1 !important;
        color: white !important;
        border-color: #0D47A1 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #F5F9FF;
        border-radius: 8px;
        font-weight: 600;
        color: #0D47A1;
    }
    
    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# COLOR PALETTE
# ============================================================
COLORS = {
    'primary': '#0D47A1',
    'primary_light': '#1565C0',
    'primary_lighter': '#1976D2',
    'accent': '#42A5F5',
    'light': '#BBDEFB',
    'lighter': '#E3F2FD',
    'lightest': '#F5F9FF',
    'text': '#1A1A2E',
    'text_muted': '#546E7A',
    'success': '#2E7D32',
    'warning': '#E65100',
    'danger': '#C62828',
    'neutral': '#78909C',
}

PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_COLOR_SEQUENCE = ['#0D47A1', '#1565C0', '#1976D2', '#42A5F5', '#64B5F6', '#90CAF9', '#BBDEFB']

# ============================================================
# DATA LOADING & PREPROCESSING (Cached)
# ============================================================
@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_data():
    """Load and preprocess the Superstore dataset."""
    df = pd.read_csv('train.csv')
    
    # Parse dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y', dayfirst=True)
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y', dayfirst=True)
    
    # Time features
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Quarter'] = df['Order Date'].dt.quarter
    df['Order Week'] = df['Order Date'].dt.isocalendar().week.astype(int)
    df['Order DayOfWeek'] = df['Order Date'].dt.dayofweek
    df['YearMonth'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
    df['YearWeek'] = df['Order Date'].dt.to_period('W').dt.to_timestamp()
    
    # Season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    df['Season'] = df['Order Month'].apply(get_season)
    
    # Shipping duration
    df['Shipping Days'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    return df

@st.cache_data(show_spinner="Computing aggregations...")
def compute_aggregations(df):
    """Pre-compute common aggregations for performance."""
    # Monthly sales
    monthly = df.groupby('YearMonth')['Sales'].sum().reset_index()
    monthly.columns = ['Date', 'Sales']
    
    # Weekly sales
    weekly = df.groupby('YearWeek')['Sales'].sum().reset_index()
    weekly.columns = ['Date', 'Sales']
    
    # Yearly sales
    yearly = df.groupby('Order Year')['Sales'].sum().reset_index()
    yearly.columns = ['Year', 'Sales']
    
    # Region sales
    region_sales = df.groupby('Region')['Sales'].sum().reset_index()
    region_sales.columns = ['Region', 'Sales']
    
    # Category sales
    cat_sales = df.groupby('Category')['Sales'].sum().reset_index()
    cat_sales.columns = ['Category', 'Sales']
    
    # Sub-category sales
    subcat_sales = df.groupby(['Category', 'Sub-Category'])['Sales'].sum().reset_index()
    
    # Monthly by category
    monthly_cat = df.groupby(['YearMonth', 'Category'])['Sales'].sum().reset_index()
    
    # Monthly by region
    monthly_reg = df.groupby(['YearMonth', 'Region'])['Sales'].sum().reset_index()
    
    return {
        'monthly': monthly,
        'weekly': weekly,
        'yearly': yearly,
        'region_sales': region_sales,
        'cat_sales': cat_sales,
        'subcat_sales': subcat_sales,
        'monthly_cat': monthly_cat,
        'monthly_reg': monthly_reg,
    }

# ============================================================
# FORECASTING FUNCTIONS (Cached)
# ============================================================
@st.cache_data(show_spinner="Training Prophet model...")
def train_prophet_model(series_data, periods=12):
    """Train Prophet model and return forecast."""
    # Prepare data for Prophet
    prophet_df = series_data.rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    # Train model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    model.fit(prophet_df)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=periods, freq='ME')
    forecast = model.predict(future)
    
    return model, forecast

@st.cache_data(show_spinner="Generating forecast...")
def get_segment_forecast(df, segment_col, segment_val, periods=12):
    """Get forecast for a specific segment."""
    if segment_col is None or segment_val == 'Overall':
        # Overall forecast
        seg_data = df.groupby('YearMonth')['Sales'].sum().reset_index()
    else:
        seg_data = df[df[segment_col] == segment_val].groupby('YearMonth')['Sales'].sum().reset_index()
    
    seg_data.columns = ['Date', 'Sales']
    
    if len(seg_data) < 12:
        return None, None, None
    
    model, forecast = train_prophet_model(seg_data, periods)
    
    # Calculate metrics on last 12 months (validation)
    train_size = len(seg_data) - 12
    if train_size > 12:
        train_data = seg_data.iloc[:train_size]
        test_data = seg_data.iloc[train_size:]
        
        model_val = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model_val.fit(train_data.rename(columns={'Date': 'ds', 'Sales': 'y'}))
        future_val = model_val.make_future_dataframe(periods=12, freq='ME')
        forecast_val = model_val.predict(future_val)
        
        y_true = test_data['Sales'].values
        y_pred = forecast_val.iloc[-12:]['yhat'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        mae = rmse = mape = None
    
    return model, forecast, {'mae': mae, 'rmse': rmse, 'mape': mape}

# ============================================================
# ANOMALY DETECTION (Cached)
# ============================================================
@st.cache_data(show_spinner="Detecting anomalies...")
def detect_anomalies(weekly_data, contamination=0.05):
    """Run Isolation Forest and Z-Score anomaly detection."""
    df = weekly_data.copy()
    df = df.set_index('Date')
    df = df.asfreq('W-MON')
    df['Sales'] = df['Sales'].interpolate()
    df = df.reset_index()
    
    # Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42)
    df['iso_anomaly'] = iso.fit_predict(df[['Sales']])
    df['iso_score'] = iso.score_samples(df[['Sales']])
    
    # Z-Score (rolling)
    window = 4
    df['rolling_mean'] = df['Sales'].rolling(window=window, center=False).mean()
    df['rolling_std'] = df['Sales'].rolling(window=window, center=False).std()
    df['z_score'] = (df['Sales'] - df['rolling_mean']) / df['rolling_std']
    df['z_anomaly'] = np.where(np.abs(df['z_score']) > 2, -1, 1)
    
    # Combined
    df['is_anomaly'] = ((df['iso_anomaly'] == -1) | (df['z_anomaly'] == -1)).astype(int)
    df['anomaly_type'] = 'Normal'
    df.loc[(df['iso_anomaly'] == -1) & (df['z_anomaly'] == -1), 'anomaly_type'] = 'Both'
    df.loc[(df['iso_anomaly'] == -1) & (df['z_anomaly'] == 1), 'anomaly_type'] = 'Isolation Forest'
    df.loc[(df['iso_anomaly'] == 1) & (df['z_anomaly'] == -1), 'anomaly_type'] = 'Z-Score'
    
    return df

# ============================================================
# CLUSTERING (Cached)
# ============================================================
@st.cache_data(show_spinner="Computing product segments...")
def compute_product_segments(df):
    """Compute K-Means clustering on product sub-categories."""
    # Feature engineering at sub-category level
    monthly_sub = df.groupby(['Sub-Category', 'Order Year', 'Order Month'])['Sales'].sum().reset_index()
    
    # Volatility (std of monthly sales)
    volatility = monthly_sub.groupby('Sub-Category')['Sales'].std().rename('Volatility')
    
    # Total sales
    total_sales = df.groupby('Sub-Category')['Sales'].sum().rename('Total_Sales')
    
    # Average order value
    avg_order = df.groupby('Sub-Category')['Sales'].mean().rename('Avg_Order_Value')
    
    # YoY Growth (last year vs previous)
    annual_sub = df.groupby(['Sub-Category', 'Order Year'])['Sales'].sum().unstack(fill_value=0)
    if annual_sub.shape[1] >= 2:
        growth = annual_sub.pct_change(axis=1).iloc[:, -1].rename('Growth_Rate')
    else:
        growth = pd.Series(0, index=annual_sub.index, name='Growth_Rate')
    
    # Combine features
    features = pd.concat([total_sales, volatility, avg_order, growth], axis=1).fillna(0)
    
    # Scale
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    # K-Means (4 clusters)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)
    
    result = features.copy()
    result['Cluster'] = clusters
    result['PCA1'] = pca_result[:, 0]
    result['PCA2'] = pca_result[:, 1]
    
    # Label clusters based on characteristics
    cluster_labels = {}
    for c in range(4):
        cluster_data = result[result['Cluster'] == c]
        avg_sales = cluster_data['Total_Sales'].mean()
        avg_vol = cluster_data['Volatility'].mean()
        avg_growth = cluster_data['Growth_Rate'].mean()
        
        if avg_sales > result['Total_Sales'].median() and avg_vol < result['Volatility'].median():
            label = "High Volume, Stable"
            badge = "badge-high"
        elif avg_sales < result['Total_Sales'].median() and avg_vol > result['Volatility'].median():
            label = "Low Volume, Volatile"
            badge = "badge-volatile"
        elif avg_growth > 0.1:
            label = "Growing Demand"
            badge = "badge-growing"
        else:
            label = "Declining Demand"
            badge = "badge-declining"
        cluster_labels[c] = {'label': label, 'badge': badge}
    
    result['Cluster_Label'] = result['Cluster'].map({k: v['label'] for k, v in cluster_labels.items()})
    result['Cluster_Badge'] = result['Cluster'].map({k: v['badge'] for k, v in cluster_labels.items()})
    
    return result, cluster_labels, scaler, kmeans, pca

# ============================================================
# PLOTLY CHART HELPERS
# ============================================================
def apply_plotly_theme(fig, height=400, show_legend=True):
    """Apply consistent styling to Plotly figures."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        font=dict(family="Inter, sans-serif", size=12, color=COLORS['text']),
        title_font=dict(size=14, color=COLORS['primary'], family="Inter, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=COLORS['text_muted']),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=COLORS['light'],
            borderwidth=1
        ) if show_legend else dict(visible=False),
        margin=dict(l=50, r=30, t=60, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS['lighter'],
            linecolor=COLORS['light'],
            tickfont=dict(size=11, color=COLORS['text_muted']),
            title_font=dict(size=12, color=COLORS['text'])
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS['lighter'],
            linecolor=COLORS['light'],
            tickfont=dict(size=11, color=COLORS['text_muted']),
            title_font=dict(size=12, color=COLORS['text'])
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor=COLORS['light']
        )
    )
    return fig

def format_currency(val):
    """Format number as currency."""
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    elif val >= 1e6:
        return f"${val/1e6:.2f}M"
    elif val >= 1e3:
        return f"${val/1e3:.2f}K"
    else:
        return f"${val:,.0f}"

def format_number(val):
    """Format number with K/M/B suffix."""
    if val >= 1e9:
        return f"{val/1e9:.2f}B"
    elif val >= 1e6:
        return f"{val/1e6:.2f}M"
    elif val >= 1e3:
        return f"{val/1e3:.2f}K"
    else:
        return f"{val:,.0f}"

# ============================================================
# SIDEBAR FILTERS
# ============================================================
def render_sidebar_filters(df):
    """Render sidebar filters and return filtered dataframe."""
    st.sidebar.markdown("### 🎛️ Filters")
    
    # Date range
    min_date = df['Order Date'].min().date()
    max_date = df['Order Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range"
    )
    
    # Region filter
    regions = ['All'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("🌍 Region", regions, key="region_filter")
    
    # Category filter
    categories = ['All'] + sorted(df['Category'].unique().tolist())
    selected_category = st.sidebar.selectbox("📦 Category", categories, key="category_filter")
    
    # Segment filter
    segments = ['All'] + sorted(df['Segment'].unique().tolist())
    selected_segment = st.sidebar.selectbox("👥 Segment", segments, key="segment_filter")
    
    # Apply filters
    filtered = df.copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered['Order Date'].dt.date >= start_date) & 
            (filtered['Order Date'].dt.date <= end_date)
        ]
    
    if selected_region != 'All':
        filtered = filtered[filtered['Region'] == selected_region]
    
    if selected_category != 'All':
        filtered = filtered[filtered['Category'] == selected_category]
    
    if selected_segment != 'All':
        filtered = filtered[filtered['Segment'] == selected_segment]
    
    # Sidebar metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    
    total_sales = filtered['Sales'].sum()
    total_orders = filtered['Order ID'].nunique()
    avg_order = total_sales / total_orders if total_orders > 0 else 0
    unique_products = filtered['Product ID'].nunique()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Sales", format_currency(total_sales))
        st.metric("Avg Order Value", format_currency(avg_order))
    with col2:
        st.metric("Total Orders", format_number(total_orders))
        st.metric("Unique Products", format_number(unique_products))
    
    return filtered, {
        'date_range': date_range,
        'region': selected_region,
        'category': selected_category,
        'segment': selected_segment
    }

# ============================================================
# KPI CARD COMPONENT
# ============================================================
def render_kpi_card(label, value, delta=None, delta_label=""):
    """Render a styled KPI card."""
    delta_html = ""
    if delta is not None:
        delta_class = "positive" if delta >= 0 else "negative"
        delta_sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="kpi-delta {delta_class}">{delta_sign}{delta:.1f}% {delta_label}</div>'
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE 1: SALES OVERVIEW
# ============================================================
def page_sales_overview(df, aggs):
    """Sales Overview Dashboard."""
    st.markdown('<div class="section-header">📊 Sales Overview Dashboard</div>', unsafe_allow_html=True)
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    
    current_year = df['Order Year'].max()
    prev_year = current_year - 1
    
    sales_current = df[df['Order Year'] == current_year]['Sales'].sum()
    sales_prev = df[df['Order Year'] == prev_year]['Sales'].sum()
    yoy_growth = ((sales_current - sales_prev) / sales_prev * 100) if sales_prev > 0 else 0
    
    orders_current = df[df['Order Year'] == current_year]['Order ID'].nunique()
    orders_prev = df[df['Order Year'] == prev_year]['Order ID'].nunique()
    orders_yoy = ((orders_current - orders_prev) / orders_prev * 100) if orders_prev > 0 else 0
    
    avg_order_current = sales_current / orders_current if orders_current > 0 else 0
    avg_order_prev = sales_prev / orders_prev if orders_prev > 0 else 0
    aov_yoy = ((avg_order_current - avg_order_prev) / avg_order_prev * 100) if avg_order_prev > 0 else 0
    
    profit_current = df[df['Order Year'] == current_year]['Profit'].sum() if 'Profit' in df.columns else 0
    profit_prev = df[df['Order Year'] == prev_year]['Profit'].sum() if 'Profit' in df.columns else 0
    profit_yoy = ((profit_current - profit_prev) / profit_prev * 100) if profit_prev > 0 else 0
    
    with col1:
        render_kpi_card("Total Sales (YoY)", format_currency(sales_current), yoy_growth, "vs Prior Year")
    with col2:
        render_kpi_card("Total Orders (YoY)", format_number(orders_current), orders_yoy, "vs Prior Year")
    with col3:
        render_kpi_card("Avg Order Value (YoY)", format_currency(avg_order_current), aov_yoy, "vs Prior Year")
    with col4:
        render_kpi_card("Total Profit (YoY)", format_currency(profit_current), profit_yoy, "vs Prior Year")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row 1: Yearly Trend + Monthly Trend
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="section-subheader">📅 Yearly Sales Trend</div>', unsafe_allow_html=True)
        yearly_data = aggs['yearly']
        fig = px.bar(
            yearly_data, x='Year', y='Sales',
            color='Sales', color_continuous_scale='Blues',
            text='Sales'
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(coloraxis_showscale=False, height=350)
        fig = apply_plotly_theme(fig, height=350, show_legend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-subheader">📈 Monthly Sales Trend</div>', unsafe_allow_html=True)
        monthly_data = aggs['monthly']
        fig = px.line(
            monthly_data, x='Date', y='Sales',
            markers=True, line_shape='spline'
        )
        fig.update_traces(line=dict(color=COLORS['primary'], width=3), marker=dict(size=6, color=COLORS['primary']))
        # Add year-over-year comparison lines
        for year in sorted(monthly_data['Date'].dt.year.unique()):
            year_data = monthly_data[monthly_data['Date'].dt.year == year]
            if year != monthly_data['Date'].dt.year.max():
                fig.add_scatter(
                    x=year_data['Date'], y=year_data['Sales'],
                    mode='lines', line=dict(color=COLORS['light'], width=1, dash='dot'),
                    name=str(year), showlegend=True, opacity=0.5
                )
        fig = apply_plotly_theme(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2: Region + Category
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-subheader">🌍 Sales by Region</div>', unsafe_allow_html=True)
        region_data = aggs['region_sales'].sort_values('Sales', ascending=True)
        fig = px.bar(
            region_data, x='Sales', y='Region', orientation='h',
            color='Sales', color_continuous_scale='Blues',
            text='Sales'
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(coloraxis_showscale=False, height=350)
        fig = apply_plotly_theme(fig, height=350, show_legend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-subheader">📦 Sales by Category</div>', unsafe_allow_html=True)
        cat_data = aggs['cat_sales']
        fig = px.pie(
            cat_data, values='Sales', names='Category',
            color_discrete_sequence=PLOTLY_COLOR_SEQUENCE,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig = apply_plotly_theme(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 3: Sub-Category + Seasonal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-subheader">📋 Sales by Sub-Category</div>', unsafe_allow_html=True)
        subcat_data = aggs['subcat_sales'].sort_values('Sales', ascending=True)
        fig = px.bar(
            subcat_data, x='Sales', y='Sub-Category', orientation='h',
            color='Category', color_discrete_sequence=PLOTLY_COLOR_SEQUENCE,
            text='Sales'
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        fig = apply_plotly_theme(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-subheader">🌤️ Seasonal Pattern</div>', unsafe_allow_html=True)
        seasonal = df.groupby('Season')['Sales'].mean().reset_index()
        season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
        seasonal['Season'] = pd.Categorical(seasonal['Season'], categories=season_order, ordered=True)
        seasonal = seasonal.sort_values('Season')
        
        fig = px.bar(
            seasonal, x='Season', y='Sales',
            color='Sales', color_continuous_scale='Blues',
            text='Sales'
        )
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(coloraxis_showscale=False, height=350)
        fig = apply_plotly_theme(fig, height=350, show_legend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Download button
    st.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================
# PAGE 2: FORECAST EXPLORER
# ============================================================
def page_forecast_explorer(df):
    """Forecast Explorer with Prophet."""
    st.markdown('<div class="section-header">🔮 Forecast Explorer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">📋 How to Use</div>
        <div class="info-box-text">
            Select a segment (Overall, Category, or Region), choose forecast horizon, and view the Prophet forecast with 95% confidence intervals. 
            Model performance metrics are calculated on a 12-month holdout period.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        segment_type = st.selectbox(
            "📊 Forecast Segment",
            ["Overall", "By Category", "By Region"],
            key="forecast_segment_type"
        )
    
    if segment_type == "By Category":
        segment_options = sorted(df['Category'].unique().tolist())
        segment_col = 'Category'
    elif segment_type == "By Region":
        segment_options = sorted(df['Region'].unique().tolist())
        segment_col = 'Region'
    else:
        segment_options = ["Overall"]
        segment_col = None
    
    with col2:
        selected_segment = st.selectbox("🎯 Select Segment", segment_options, key="forecast_segment")
    
    with col3:
        horizon = st.slider("📅 Forecast Horizon (Months)", 1, 24, 12, key="forecast_horizon")
    
    # Prepare data
    if segment_type == "Overall":
        series_data = df.groupby('YearMonth')['Sales'].sum().reset_index()
        series_data.columns = ['Date', 'Sales']
        segment_name = "Overall Sales"
    else:
        seg_df = df[df[segment_col] == selected_segment]
        series_data = seg_df.groupby('YearMonth')['Sales'].sum().reset_index()
        series_data.columns = ['Date', 'Sales']
        segment_name = f"{segment_type}: {selected_segment}"
    
    if len(series_data) < 12:
        st.warning("⚠️ Insufficient data for forecasting (need at least 12 months).")
        return
    
    # Train model and forecast
    with st.spinner(f"Training Prophet model for {segment_name}..."):
        model, forecast, metrics = get_segment_forecast(
            df, segment_col, selected_segment if segment_col else 'Overall', periods=horizon
        )
    
    if model is None:
        st.error("Could not generate forecast for this segment.")
        return
    
    # Split historical vs forecast
    last_hist_date = series_data['Date'].max()
    hist_forecast = forecast[forecast['ds'] <= last_hist_date]
    future_forecast = forecast[forecast['ds'] > last_hist_date]
    
    # ===== MAIN FORECAST CHART =====
    st.markdown('<div class="section-subheader">📈 Forecast Visualization</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Historical actuals
    fig.add_trace(go.Scatter(
        x=series_data['Date'], y=series_data['Sales'],
        mode='lines+markers', name='Historical Actuals',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=5, color=COLORS['primary']),
        hovertemplate='%{x|%b %Y}: $%{y:,.0f}<extra></extra>'
    ))
    
    # Historical fitted (in-sample)
    fig.add_trace(go.Scatter(
        x=hist_forecast['ds'], y=hist_forecast['yhat'],
        mode='lines', name='Fitted (In-Sample)',
        line=dict(color=COLORS['accent'], width=2, dash='dot'),
        hovertemplate='%{x|%b %Y}: $%{y:,.0f}<extra></extra>'
    ))
    
    # Future forecast
    fig.add_trace(go.Scatter(
        x=future_forecast['ds'], y=future_forecast['yhat'],
        mode='lines+markers', name='Forecast',
        line=dict(color=COLORS['primary_light'], width=3),
        marker=dict(size=6, color=COLORS['primary_light']),
        hovertemplate='%{x|%b %Y}: $%{y:,.0f}<extra></extra>'
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([future_forecast['ds'], future_forecast['ds'][::-1]]),
        y=pd.concat([future_forecast['yhat_upper'], future_forecast['yhat_lower'][::-1]]),
        fill='toself', fillcolor=f'rgba(21, 101, 192, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))
    
    # Forecast boundary. Use a shape plus annotation instead of add_vline because
    # Plotly's annotation helper cannot average pandas Timestamp values.
    fig.add_shape(
        type="line",
        x0=last_hist_date,
        x1=last_hist_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color=COLORS['neutral'], dash="dash")
    )
    fig.add_annotation(
        x=last_hist_date,
        y=1,
        xref="x",
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        yshift=12,
        font=dict(size=11, color=COLORS['text_muted'])
    )
    
    fig = apply_plotly_theme(fig, height=500)
    fig.update_layout(
        yaxis_title="Sales ($)",
        xaxis_title="Date",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ===== METRICS =====
    if metrics and metrics['mae'] is not None:
        st.markdown('<div class="section-subheader">📊 Model Performance (12-Month Holdout)</div>', unsafe_allow_html=True)
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            render_kpi_card("MAE", format_currency(metrics['mae']))
        with mcol2:
            render_kpi_card("RMSE", format_currency(metrics['rmse']))
        with mcol3:
            render_kpi_card("MAPE", f"{metrics['mape']:.1f}%")
    
    # ===== FORECAST TABLE =====
    st.markdown('<div class="section-subheader">📋 Forecast Details</div>', unsafe_allow_html=True)
    
    forecast_display = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_display.columns = ['Month', 'Forecast', 'Lower Bound (95%)', 'Upper Bound (95%)']
    forecast_display['Month'] = forecast_display['Month'].dt.strftime('%b %Y')
    for col in ['Forecast', 'Lower Bound (95%)', 'Upper Bound (95%)']:
        forecast_display[col] = forecast_display[col].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(forecast_display, use_container_width=True, hide_index=True)
    
    # Download forecast
    csv = forecast_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast (CSV)",
        data=csv,
        file_name=f"forecast_{segment_name.replace(' ', '_').replace(':', '')}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # ===== PROPHET COMPONENTS =====
    with st.expander("🔍 View Trend & Seasonality Components"):
        fig_comp = model.plot_components(forecast)
        st.pyplot(fig_comp)

# ============================================================
# PAGE 3: ANOMALY REPORT
# ============================================================
def page_anomaly_report(df):
    """Anomaly Detection Report."""
    st.markdown('<div class="section-header">🚨 Anomaly Detection Report</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">🔍 Detection Methods</div>
        <div class="info-box-text">
            <strong>Isolation Forest:</strong> Detects global anomalies by isolating observations in random trees. 
            <strong>Rolling Z-Score:</strong> Detects local anomalies using 4-week rolling mean/std (threshold: ±2σ).
            <strong>Both:</strong> Weeks flagged by both methods are high-confidence anomalies.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        method_filter = st.selectbox(
            "🔎 Filter by Method",
            ["All", "Both Methods", "Isolation Forest Only", "Z-Score Only"],
            key="anomaly_method"
        )
    with col2:
        contamination = st.slider("🎯 Isolation Forest Sensitivity", 0.01, 0.15, 0.05, 0.01, key="anomaly_contamination")
    
    # Weekly aggregation
    weekly = df.groupby('YearWeek')['Sales'].sum().reset_index()
    weekly.columns = ['Date', 'Sales']
    
    # Detect anomalies
    anomaly_df = detect_anomalies(weekly, contamination=contamination)
    
    # Filter
    if method_filter == "Both Methods":
        display_df = anomaly_df[anomaly_df['anomaly_type'] == 'Both']
    elif method_filter == "Isolation Forest Only":
        display_df = anomaly_df[anomaly_df['anomaly_type'] == 'Isolation Forest']
    elif method_filter == "Z-Score Only":
        display_df = anomaly_df[anomaly_df['anomaly_type'] == 'Z-Score']
    else:
        display_df = anomaly_df[anomaly_df['is_anomaly'] == 1]
    
    # ===== SUMMARY METRICS =====
    total_anomalies = len(anomaly_df[anomaly_df['is_anomaly'] == 1])
    iso_count = len(anomaly_df[anomaly_df['iso_anomaly'] == -1])
    z_count = len(anomaly_df[anomaly_df['z_anomaly'] == -1])
    both_count = len(anomaly_df[anomaly_df['anomaly_type'] == 'Both'])
    
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        render_kpi_card("Total Anomalies", str(total_anomalies))
    with mcol2:
        render_kpi_card("Isolation Forest", str(iso_count))
    with mcol3:
        render_kpi_card("Z-Score (±2σ)", str(z_count))
    with mcol4:
        render_kpi_card("Both Methods", str(both_count))
    
    # ===== ANOMALY TIME SERIES CHART =====
    st.markdown('<div class="section-subheader">📈 Weekly Sales with Anomalies</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Normal weeks
    normal = anomaly_df[anomaly_df['is_anomaly'] == 0]
    fig.add_trace(go.Scatter(
        x=normal['Date'], y=normal['Sales'],
        mode='lines+markers', name='Normal',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=4, color=COLORS['primary']),
        hovertemplate='%{x|%b %d, %Y}: $%{y:,.0f}<extra></extra>'
    ))
    
    # Anomalies by type
    colors_map = {
        'Both': COLORS['danger'],
        'Isolation Forest': COLORS['warning'],
        'Z-Score': COLORS['primary_light']
    }
    symbols_map = {
        'Both': 'diamond',
        'Isolation Forest': 'triangle-up',
        'Z-Score': 'circle'
    }
    
    for a_type in ['Both', 'Isolation Forest', 'Z-Score']:
        subset = anomaly_df[anomaly_df['anomaly_type'] == a_type]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset['Date'], y=subset['Sales'],
                mode='markers', name=a_type,
                marker=dict(
                    size=12, color=colors_map[a_type],
                    symbol=symbols_map[a_type],
                    line=dict(width=2, color='white')
                ),
                hovertemplate=f'<b>{a_type}</b><br>%{{x|%b %d, %Y}}: $%{{y:,.0f}}<extra></extra>'
            ))
    
    fig = apply_plotly_theme(fig, height=500)
    fig.update_layout(
        yaxis_title="Weekly Sales ($)",
        xaxis_title="Week",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ===== ANOMALY DETAILS TABLE =====
    st.markdown('<div class="section-subheader">📋 Anomaly Details</div>', unsafe_allow_html=True)
    
    if len(display_df) > 0:
        table_df = display_df[['Date', 'Sales', 'anomaly_type', 'iso_score', 'z_score']].copy()
        table_df.columns = ['Week', 'Sales', 'Detection Method', 'Isolation Score', 'Z-Score']
        table_df['Week'] = table_df['Week'].dt.strftime('%b %d, %Y')
        table_df['Sales'] = table_df['Sales'].apply(lambda x: f"${x:,.0f}")
        table_df['Isolation Score'] = table_df['Isolation Score'].apply(lambda x: f"{x:.3f}")
        table_df['Z-Score'] = table_df['Z-Score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        # Add business context
        def get_context(row):
            week = pd.to_datetime(row['Week'])
            month = week.month
            if month in [11, 12]:
                return "🛍️ Holiday Season / Black Friday"
            elif month == 1:
                return "📉 Post-Holiday Slowdown"
            elif month in [3, 6, 9]:
                return "📊 Quarter-End Push"
            elif month in [7, 8]:
                return "☀️ Summer Season"
            else:
                return "📦 Regular Period"
        
        table_df['Business Context'] = table_df.apply(get_context, axis=1)
        
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        
        # Download
        csv = table_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Anomaly Report (CSV)",
            data=csv,
            file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No anomalies match the selected filter.")
    
    # ===== METHOD COMPARISON =====
    with st.expander("📊 Method Comparison Analysis"):
        st.markdown(f"""
        **Isolation Forest** detected **{iso_count}** anomalous weeks globally.
        
        **Rolling Z-Score (±2σ)** detected **{z_count}** anomalous weeks locally.
        
        **Overlap (Both Methods):** **{both_count}** weeks — these are high-confidence anomalies.
        
        **Interpretation:**
        - **Isolation Forest** excels at finding extreme outliers across the entire history (e.g., Black Friday spikes).
        - **Rolling Z-Score** catches sudden shifts relative to recent trends (e.g., unexpected dip after a promo).
        - **Agreement between both** signals truly exceptional events requiring investigation.
        """)

# ============================================================
# PAGE 4: PRODUCT DEMAND SEGMENTS
# ============================================================
def page_product_segments(df):
    """Product Demand Segmentation with K-Means."""
    st.markdown('<div class="section-header">🎯 Product Demand Segmentation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">📊 Segmentation Methodology</div>
        <div class="info-box-text">
            Products are clustered using K-Means (k=4) on four features: <strong>Total Sales Volume</strong>, 
            <strong>Sales Volatility (monthly std)</strong>, <strong>Average Order Value</strong>, and <strong>YoY Growth Rate</strong>. 
            Features are standardized before clustering. PCA is used for 2D visualization.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Compute segments
    segments_df, cluster_labels, scaler, kmeans, pca = compute_product_segments(df)
    
    # ===== PCA SCATTER PLOT =====
    st.markdown('<div class="section-subheader">📍 Segment Visualization (PCA Projection)</div>', unsafe_allow_html=True)
    
    fig = px.scatter(
        segments_df, x='PCA1', y='PCA2',
        color='Cluster_Label', symbol='Cluster_Label',
        hover_data=['Total_Sales', 'Volatility', 'Growth_Rate', 'Avg_Order_Value'],
        color_discrete_sequence=PLOTLY_COLOR_SEQUENCE,
        size='Total_Sales', size_max=30,
        labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'}
    )
    
    # Add labels
    for idx, row in segments_df.iterrows():
        fig.add_annotation(
            x=row['PCA1'], y=row['PCA2'],
            text=idx, showarrow=False,
            font=dict(size=9, color=COLORS['text']),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=COLORS['light'],
            borderwidth=1,
            borderpad=2
        )
    
    fig = apply_plotly_theme(fig, height=550)
    fig.update_layout(legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
    st.plotly_chart(fig, use_container_width=True)
    
    # ===== CLUSTER PROFILE TABLE =====
    st.markdown('<div class="section-subheader">📋 Segment Profiles</div>', unsafe_allow_html=True)
    
    profile_cols = ['Cluster', 'Cluster_Label', 'Total_Sales', 'Volatility', 'Growth_Rate', 'Avg_Order_Value']
    display_df = segments_df[profile_cols].copy()
    display_df.columns = ['Cluster ID', 'Segment', 'Total Sales', 'Volatility (Std)', 'YoY Growth', 'Avg Order Value']
    display_df['Total Sales'] = display_df['Total Sales'].apply(format_currency)
    display_df['Volatility (Std)'] = display_df['Volatility (Std)'].apply(lambda x: f"${x:,.0f}")
    display_df['YoY Growth'] = display_df['YoY Growth'].apply(lambda x: f"{x*100:.1f}%")
    display_df['Avg Order Value'] = display_df['Avg Order Value'].apply(format_currency)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # ===== STRATEGY CARDS =====
    st.markdown('<div class="section-subheader">💡 Recommended Stocking Strategies</div>', unsafe_allow_html=True)
    
    strategy_details = {
        "High Volume, Stable": {
            "icon": "🏆",
            "title": "Cash Cows — High Volume, Stable Demand",
            "strategy": "Maintain high safety stock levels. Implement automated reorder points. Negotiate volume discounts with suppliers. These are your revenue backbone — prioritize 99%+ service levels.",
            "badge": "badge-high"
        },
        "Low Volume, Volatile": {
            "icon": "⚠️",
            "title": "Niche/Risky — Low Volume, High Volatility",
            "strategy": "Adopt Just-in-Time (JIT) replenishment. Avoid bulk purchasing. Use drop-shipping where possible. Set higher reorder points but lower order quantities. Monitor closely for demand signals.",
            "badge": "badge-volatile"
        },
        "Growing Demand": {
            "icon": "🚀",
            "title": "Stars — Growing Demand",
            "strategy": "Increase safety stock buffers by 20-30%. Expedite supplier lead times. Invest in demand sensing. Stock-outs here mean disproportionate revenue loss. Consider dual-sourcing critical SKUs.",
            "badge": "badge-growing"
        },
        "Declining Demand": {
            "icon": "📉",
            "title": "Laggards — Declining Demand",
            "strategy": "Reduce reorder quantities. Run clearance promotions. Negotiate return agreements with suppliers. Phase out slow movers. Redirect shelf space and capital to Growing/High Volume segments.",
            "badge": "badge-declining"
        }
    }
    
    for cluster_id in sorted(segments_df['Cluster'].unique()):
        cluster_info = segments_df[segments_df['Cluster'] == cluster_id]
        label = cluster_info['Cluster_Label'].iloc[0]
        badge = cluster_info['Cluster_Badge'].iloc[0]
        subcats = cluster_info.index.tolist()
        
        detail = strategy_details.get(label, strategy_details["High Volume, Stable"])
        
        st.markdown(f"""
        <div class="strategy-card">
            <span class="strategy-badge {detail['badge']}">{detail['icon']} {label}</span>
            <h4>{detail['title']}</h4>
            <p><strong>Sub-Categories:</strong> {', '.join(subcats)}</p>
            <p>{detail['strategy']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== CLUSTER SUMMARY METRICS =====
    st.markdown('<div class="section-subheader">📊 Cluster Summary Statistics</div>', unsafe_allow_html=True)
    
    summary = segments_df.groupby('Cluster_Label').agg(
        Sub_Categories=('Cluster_Label', 'count'),
        Avg_Total_Sales=('Total_Sales', 'mean'),
        Avg_Volatility=('Volatility', 'mean'),
        Avg_Growth=('Growth_Rate', 'mean'),
        Avg_Order_Value=('Avg_Order_Value', 'mean')
    ).round(2)
    
    summary.columns = ['# Sub-Categories', 'Avg Total Sales', 'Avg Volatility', 'Avg YoY Growth', 'Avg Order Value']
    summary['Avg Total Sales'] = summary['Avg Total Sales'].apply(format_currency)
    summary['Avg Volatility'] = summary['Avg Volatility'].apply(lambda x: f"${x:,.0f}")
    summary['Avg YoY Growth'] = summary['Avg YoY Growth'].apply(lambda x: f"{x*100:.1f}%")
    summary['Avg Order Value'] = summary['Avg Order Value'].apply(format_currency)
    
    st.dataframe(summary, use_container_width=True)
    
    # Download
    csv = segments_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Segmentation Data (CSV)",
        data=csv,
        file_name=f"product_segments_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Load data
    df = load_data()
    aggs = compute_aggregations(df)
    
    # Sidebar filters
    filtered_df, filter_state = render_sidebar_filters(df)
    
    # Recompute aggregations for filtered data
    if len(filtered_df) != len(df):
        filtered_aggs = compute_aggregations(filtered_df)
    else:
        filtered_aggs = aggs
    
    # Header
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
        <div>
            <h1 style="margin: 0; color: #0D47A1; font-size: 2rem; font-weight: 700;">Sales Forecasting & Demand Intelligence</h1>
            <p style="margin: 0.25rem 0 0 0; color: #546E7A; font-size: 1rem;">End-to-end demand planning: forecasting, anomaly detection & product segmentation</p>
        </div>
        <div style="text-align: right; color: #546E7A; font-size: 0.85rem;">
            Data: {min_date} → {max_date} | {n_records:,} records | {n_products:,} products
        </div>
    </div>
    """.format(
        min_date=filtered_df['Order Date'].min().strftime('%b %Y'),
        max_date=filtered_df['Order Date'].max().strftime('%b %Y'),
        n_records=len(filtered_df),
        n_products=filtered_df['Product ID'].nunique()
    ), unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Sales Overview",
        "🔮 Forecast Explorer",
        "🚨 Anomaly Report",
        "🎯 Product Segments"
    ])
    
    with tab1:
        page_sales_overview(filtered_df, filtered_aggs)
    
    with tab2:
        page_forecast_explorer(filtered_df)
    
    with tab3:
        page_anomaly_report(filtered_df)
    
    with tab4:
        page_product_segments(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #546E7A; font-size: 0.85rem; padding: 1rem;">
        Built with Streamlit • Prophet • Plotly • Scikit-Learn | 
        Superstore Sales Dataset (Kaggle) | 
        Internship Project — Week 3 & 4
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
