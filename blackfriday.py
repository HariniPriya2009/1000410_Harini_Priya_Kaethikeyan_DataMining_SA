import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy import stats
from mlxtend.frequent_patterns import apriori, association_rules

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Black Friday Sales Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
# MODERN DARK THEME CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1f 100%);
    }
    
    /* Glassmorphism Header */
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    .main-header h1 {
        background: linear-gradient(135deg, #00f5d4 0%, #00bbf9 50%, #9b5de5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Neon Accent Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        border-radius: 16px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 245, 212, 0.3);
        box-shadow: 0 10px 40px rgba(0, 245, 212, 0.15);
    }
    
    /* Insight Boxes with Neon Glow */
    .insight-box {
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.15) 0%, rgba(155, 93, 229, 0.15) 100%);
        border: 1px solid rgba(0, 245, 212, 0.3);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 245, 212, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(255, 159, 67, 0.15) 100%);
        border: 1px solid rgba(255, 107, 107, 0.3);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(0, 187, 249, 0.15) 0%, rgba(0, 245, 212, 0.15) 100%);
        border: 1px solid rgba(0, 187, 249, 0.3);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.2rem;
        border-radius: 16px;
    }
    
    div[data-testid="stMetric"] label {
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="stMetric"] > div > div {
        background: linear-gradient(135deg, #00f5d4 0%, #00bbf9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 24px;
        padding-right: 24px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px 12px 0 0;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.6);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.08);
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.2) 0%, rgba(155, 93, 229, 0.2) 100%);
        border: 1px solid rgba(0, 245, 212, 0.4);
        color: #00f5d4 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 26, 62, 0.95) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    section[data-testid="stSidebar"] .element-container {
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* Headings */
    h2 {
        background: linear-gradient(135deg, #00f5d4 0%, #00bbf9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
    }
    
    h3 {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 600 !important;
    }
    
    h4 {
        color: rgba(255, 255, 255, 0.85) !important;
        font-weight: 600 !important;
    }
    
    /* DataFrame Styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
    }
    
    /* Sliders and Inputs */
    .stSlider, .stSelectbox {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 8px;
    }
    
    /* Plot Container */
    .js-plotly-plot, .matplotlib-figure {
        background: transparent !important;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00f5d4 0%, #9b5de5 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00bbf9 0%, #9b5de5 100%);
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Markdown Text */
    p, li {
        color: rgba(255, 255, 255, 0.75);
    }
    
    /* Info/Warning Boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("BlackFriday.csv")
    
    # Data cleaning
    df['Product_Category_2'].fillna(0, inplace=True)
    df['Product_Category_3'].fillna(0, inplace=True)
    
    # Encode categorical variables
    df['Gender_Encoded'] = df['Gender'].map({'M': 0, 'F': 1})
    
    age_mapping = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, 
                   '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Encoded'] = df['Age'].map(age_mapping)
    
    df['City_Encoded'] = df['City_Category'].astype('category').cat.codes
    
    stay_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4}
    df['Stay_Encoded'] = df['Stay_In_Current_City_Years'].map(stay_mapping)
    
    return df

# Try to load data
try:
    df = load_data()
    data_loaded = True
except:
    data_loaded = False

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: #00f5d4; margin-bottom: 0.5rem;">🥷 Data Ninja</h2>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">Your Analytics Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎛️ Control Panel")
    
    if data_loaded:
        st.markdown(f"""
        <div style="background: rgba(0, 245, 212, 0.1); padding: 1rem; border-radius: 12px; border: 1px solid rgba(0, 245, 212, 0.2);">
            <p style="color: rgba(255,255,255,0.8); margin: 0;"><b>📊 Dataset Stats</b></p>
            <p style="color: rgba(255,255,255,0.6); font-size: 0.85rem; margin: 0.5rem 0 0 0;">
                • {len(df):,} transactions<br>
                • {df['User_ID'].nunique():,} unique users<br>
                • {df['Product_ID'].nunique():,} products
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Analysis Settings")
    
    n_clusters = st.slider("Number of Clusters", 2, 8, 4)
    anomaly_rate = st.slider("Anomaly Sensitivity", 0.01, 0.10, 0.02)
    
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(155, 93, 229, 0.1); padding: 1rem; border-radius: 12px; border: 1px solid rgba(155, 93, 229, 0.2);">
        <p style="color: rgba(255,255,255,0.8); margin: 0;"><b>📌 Quick Tips</b></p>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin: 0.5rem 0 0 0;">
            • Use tabs to navigate<br>
            • Hover over charts for details<br>
            • Adjust sliders above
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════
if data_loaded:
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛍️ Black Friday Sales Analytics</h1>
        <p>Uncover hidden patterns in retail data with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row
    st.markdown("### 📈 Key Metrics at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Revenue", f"${df['Purchase'].sum():,.0f}")
    with col2:
        st.metric("🛒 Avg Purchase", f"${df['Purchase'].mean():,.2f}")
    with col3:
        st.metric("👥 Total Customers", f"{df['User_ID'].nunique():,}")
    with col4:
        st.metric("📦 Transactions", f"{len(df):,}")
    
    st.markdown("---")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🎯 Customer Segments", "🔗 Product Insights", 
        "🚨 Anomaly Detection", "💡 Recommendations"
    ])
    
    # Set dark theme for matplotlib
    plt.style.use('dark_background')
    
    # ═════════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ═════════════════════════════════════════════════════════════
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💵 Purchase Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            sns.histplot(df['Purchase'], bins=40, kde=True, color='#00f5d4', ax=ax, alpha=0.7)
            ax.axvline(df['Purchase'].mean(), color='#ff6b6b', linestyle='--', linewidth=2, label=f'Mean: ${df["Purchase"].mean():.0f}')
            ax.legend(facecolor='#1a1a3e', edgecolor='rgba(255,255,255,0.1)')
            ax.set_xlabel('Purchase Amount ($)', color='white')
            ax.set_ylabel('Count', color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("#### 🎂 Purchase by Age Group")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            age_order = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']
            palette = ['#00f5d4', '#00bbf9', '#9b5de5', '#f15bb5', '#fee440', '#ff6b6b', '#8ac926']
            sns.boxplot(x='Age', y='Purchase', data=df, order=age_order, palette=palette, ax=ax)
            ax.set_xlabel('Age Group', color='white')
            ax.set_ylabel('Purchase ($)', color='white')
            ax.tick_params(colors='white')
            plt.xticks(rotation=45)
            for spine in ax.spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 👫 Gender Analysis")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax in axes:
                ax.set_facecolor('#0f0f23')
            fig.patch.set_facecolor('#0f0f23')
            
            # Pie chart
            gender_counts = df['Gender'].value_counts()
            colors = ['#00f5d4', '#f15bb5']
            axes[0].pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', 
                       colors=colors, explode=(0.05, 0.05), shadow=True,
                       textprops={'color': 'white'})
            axes[0].set_title('Gender Distribution', color='white')
            
            # Bar chart
            gender_purchase = df.groupby('Gender')['Purchase'].mean()
            sns.barplot(x=['Male', 'Female'], y=gender_purchase.values, palette=['#00f5d4', '#f15bb5'], ax=axes[1])
            axes[1].set_title('Avg Purchase by Gender', color='white')
            axes[1].set_ylabel('Avg Purchase ($)', color='white')
            axes[1].tick_params(colors='white')
            for spine in axes[1].spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("#### 🏙️ City Category Performance")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            city_stats = df.groupby('City_Category')['Purchase'].agg(['mean', 'count'])
            city_stats.columns = ['Avg Purchase', 'Transactions']
            city_stats = city_stats.reset_index()
            
            x = np.arange(len(city_stats))
            width = 0.35
            bars1 = ax.bar(x - width/2, city_stats['Avg Purchase'], width, label='Avg Purchase', color='#00f5d4')
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, city_stats['Transactions']/100, width, label='Transactions (100s)', color='#9b5de5')
            
            ax.set_xlabel('City Category', color='white')
            ax.set_ylabel('Avg Purchase ($)', color='#00f5d4')
            ax2.set_ylabel('Transactions (100s)', color='#9b5de5')
            ax.set_xticks(x)
            ax.set_xticklabels(city_stats['City_Category'])
            ax.tick_params(colors='white')
            ax2.tick_params(colors='#9b5de5')
            ax.legend(loc='upper left', facecolor='#1a1a3e', edgecolor='rgba(255,255,255,0.1)')
            ax2.legend(loc='upper right', facecolor='#1a1a3e', edgecolor='rgba(255,255,255,0.1)')
            for spine in ax.spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Correlation Heatmap
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        numeric_cols = ['Gender_Encoded', 'Age_Encoded', 'Occupation', 'City_Encoded', 
                        'Stay_Encoded', 'Marital_Status', 'Purchase']
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#0f0f23')
        ax.set_facecolor('#0f0f23')
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', linewidths=0.5, ax=ax,
                    cbar_kws={'label': 'Correlation'})
        ax.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig)
    
    # ═════════════════════════════════════════════════════════════
    # TAB 2: CUSTOMER SEGMENTS
    # ═════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### 🎯 Customer Segmentation with K-Means Clustering")
        
        # Prepare features
        features = df[['Age_Encoded', 'Occupation', 'Marital_Status', 'Purchase']].dropna()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Elbow Method
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("##### 📐 Elbow Method Analysis")
            inertias = []
            K_range = range(1, 11)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_features)
                inertias.append(kmeans.inertia_)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            ax.plot(K_range, inertias, 'o-', linewidth=2, markersize=10, color='#00f5d4')
            ax.axvline(n_clusters, color='#ff6b6b', linestyle='--', linewidth=2, label=f'Selected K={n_clusters}')
            ax.set_xlabel('Number of Clusters (K)', color='white')
            ax.set_ylabel('Inertia', color='white')
            ax.set_title('Elbow Method for Optimal K', color='white')
            ax.legend(facecolor='#1a1a3e', edgecolor='rgba(255,255,255,0.1)')
            ax.grid(True, alpha=0.2, color='rgba(255,255,255,0.1)')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(scaled_features)
        
        with col2:
            st.markdown("##### 🎨 Cluster Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            scatter = sns.scatterplot(x=features.iloc[:, 0], y=features.iloc[:, 3], 
                                       hue=df['Cluster'], palette='Set2', s=60, alpha=0.7, ax=ax)
            ax.set_xlabel('Age (Encoded)', color='white')
            ax.set_ylabel('Purchase Amount ($)', color='white')
            ax.set_title(f'Customer Clusters (K={n_clusters})', color='white')
            ax.tick_params(colors='white')
            ax.legend(facecolor='#1a1a3e', edgecolor='rgba(255,255,255,0.1)')
            for spine in ax.spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Cluster Summary
        st.markdown("##### 📊 Cluster Summary Statistics")
        cluster_summary = df.groupby('Cluster').agg({
            'Purchase': ['mean', 'count'],
            'Age': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A',
            'Gender': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
        }).round(2)
        cluster_summary.columns = ['Avg Purchase ($)', 'Customers', 'Top Age Group', 'Top Gender']
        cluster_summary['% of Total'] = (cluster_summary['Customers'] / len(df) * 100).round(1)
        
        st.dataframe(cluster_summary.style.background_gradient(cmap='viridis'), use_container_width=True)
        
        # Cluster interpretation
        st.markdown("##### 💡 Cluster Interpretation")
        interpretations = []
        for i in range(n_clusters):
            avg_purch = df[df['Cluster'] == i]['Purchase'].mean()
            if avg_purch > df['Purchase'].quantile(0.75):
                label = "🤑 **Premium Spenders**"
            elif avg_purch > df['Purchase'].quantile(0.5):
                label = "🛍️ **Regular Shoppers**"
            else:
                label = "💰 **Budget Conscious**"
            interpretations.append(f"**Cluster {i}:** {label} - Avg: ${avg_purch:,.0f}")
        
        for interp in interpretations:
            st.markdown(f"- {interp}")
    
    # ═════════════════════════════════════════════════════════════
    # TAB 3: PRODUCT INSIGHTS
    # ═════════════════════════════════════════════════════════════
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📦 Product Category Popularity")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            cat_counts = df['Product_Category_1'].value_counts().head(10)
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(cat_counts)))
            bars = ax.barh(range(len(cat_counts)), cat_counts.values, color=colors)
            ax.set_yticks(range(len(cat_counts)))
            ax.set_yticklabels([f'Cat {c}' for c in cat_counts.index])
            ax.set_xlabel('Number of Purchases', color='white')
            ax.set_title('Top 10 Product Categories', color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            for i, v in enumerate(cat_counts.values):
                ax.text(v + 1000, i, f'{v:,}', va='center', fontsize=9, color='white')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("#### 💵 Revenue by Category")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            cat_revenue = df.groupby('Product_Category_1')['Purchase'].sum().sort_values(ascending=False).head(10)
            colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(cat_revenue)))
            ax.barh(range(len(cat_revenue)), cat_revenue.values, color=colors)
            ax.set_yticks(range(len(cat_revenue)))
            ax.set_yticklabels([f'Cat {c}' for c in cat_revenue.index])
            ax.set_xlabel('Total Revenue ($)', color='white')
            ax.set_title('Top 10 Categories by Revenue', color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 🔗 Product Association Rules")
            
            # Prepare basket
            basket = df[['Product_Category_1', 'Product_Category_2', 'Product_Category_3']].copy()
            basket = basket.astype(str)
            basket_encoded = pd.get_dummies(basket)
            
            try:
                frequent_items = apriori(basket_encoded, min_support=0.02, use_colnames=True)
                if len(frequent_items) > 0:
                    rules = association_rules(frequent_items, metric="lift", min_threshold=1.0, 
                                             num_itemsets=len(frequent_items))
                    rules = rules[rules['lift'] >= 1.2].sort_values('lift', ascending=False)
                    
                    if len(rules) > 0:
                        st.markdown("##### 🎯 Top Product Associations")
                        for idx, row in rules.head(5).iterrows():
                            ant = ', '.join([x.split('_')[-1] for x in list(row['antecedents'])])
                            con = ', '.join([x.split('_')[-1] for x in list(row['consequents'])])
                            st.markdown(f"""
                            <div class="insight-box">
                                <b>📦 Categories {ant}</b> → <b>🛒 Categories {con}</b><br>
                                <small>Lift: {row['lift']:.2f} | Confidence: {row['confidence']:.2%}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No strong association rules found. Try different parameters.")
                else:
                    st.info("No frequent itemsets found.")
            except Exception as e:
                st.warning(f"Association analysis requires more data diversity.")
        
        # Category performance by demographic
        st.markdown("#### 👫 Category Preferences by Gender")
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('#0f0f23')
        ax.set_facecolor('#0f0f23')
        gender_cat = df.groupby(['Gender', 'Product_Category_1'])['Purchase'].mean().unstack().T
        gender_cat = gender_cat.head(10)
        x = np.arange(len(gender_cat))
        width = 0.35
        ax.bar(x - width/2, gender_cat['M'], width, label='Male', color='#00f5d4')
        ax.bar(x + width/2, gender_cat['F'], width, label='Female', color='#f15bb5')
        ax.set_xlabel('Product Category', color='white')
        ax.set_ylabel('Average Purchase ($)', color='white')
        ax.set_title('Avg Purchase by Category & Gender', color='white')
        ax.set_xticks(x)
        ax.set_xticklabels(gender_cat.index)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1a3e', edgecolor='rgba(255,255,255,0.1)')
        for spine in ax.spines.values():
            spine.set_color('rgba(255,255,255,0.1)')
        plt.tight_layout()
        st.pyplot(fig)
    
    # ═════════════════════════════════════════════════════════════
    # TAB 4: ANOMALY DETECTION
    # ═════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("#### 🚨 Detecting Unusual Shopping Behavior")
        
        # Isolation Forest
        iso = IsolationForest(contamination=anomaly_rate, random_state=42)
        df['Anomaly'] = iso.fit_predict(df[['Purchase']])
        df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})
        
        anomalies = df[df['Anomaly'] == 1]
        normal = df[df['Anomaly'] == 0]
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚨 Anomalies Found", f"{len(anomalies):,}")
        with col2:
            st.metric("💰 Anomaly Avg Purchase", f"${anomalies['Purchase'].mean():,.0f}")
        with col3:
            st.metric("📊 Detection Rate", f"{len(anomalies)/len(df)*100:.2f}%")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔍 Anomaly Detection Visualization")
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            ax.scatter(normal.index[:5000], normal['Purchase'].iloc[:5000], 
                      c='#00f5d4', alpha=0.3, s=10, label='Normal')
            ax.scatter(anomalies.index, anomalies['Purchase'], 
                      c='#ff6b6b', alpha=0.8, s=50, label='Anomaly', marker='X')
            ax.axhline(df['Purchase'].mean() + 2*df['Purchase'].std(), 
                      color='#fee440', linestyle='--', label='Threshold')
            ax.set_xlabel('Transaction Index', color='white')
            ax.set_ylabel('Purchase Amount ($)', color='white')
            ax.set_title('Anomaly Detection Results', color='white')
            ax.legend(facecolor='#1a1a3e', edgecolor='rgba(255,255,255,0.1)')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("##### 📊 Anomaly Distribution")
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            sns.histplot(df['Purchase'], bins=50, color='#00f5d4', alpha=0.5, label='All', ax=ax)
            sns.histplot(anomalies['Purchase'], bins=20, color='#ff6b6b', label='Anomalies', ax=ax)
            ax.set_xlabel('Purchase Amount ($)', color='white')
            ax.set_title('Purchase Distribution with Anomalies Highlighted', color='white')
            ax.legend(facecolor='#1a1a3e', edgecolor='rgba(255,255,255,0.1)')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Anomaly details
        st.markdown("##### 📋 Top Unusual High Spenders")
        st.dataframe(anomalies[['User_ID', 'Product_ID', 'Purchase', 'Age', 'Gender', 'Occupation']]
                    .sort_values('Purchase', ascending=False).head(15)
                    .style.background_gradient(cmap='Reds', subset=['Purchase']),
                    use_container_width=True)
        
        # Anomaly demographics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 👫 Anomalies by Gender")
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            anom_gender = anomalies['Gender'].value_counts()
            ax.pie(anom_gender, labels=['Male', 'Female'], autopct='%1.1f%%', 
                  colors=['#00f5d4', '#f15bb5'], explode=(0.05, 0.05),
                  textprops={'color': 'white'})
            ax.set_title('Anomaly Distribution by Gender', color='white')
            st.pyplot(fig)
        
        with col2:
            st.markdown("##### 🎂 Anomalies by Age")
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f0f23')
            ax.set_facecolor('#0f0f23')
            sns.countplot(x='Age', data=anomalies, palette='viridis', 
                         order=['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'], ax=ax)
            ax.set_title('Anomaly Distribution by Age Group', color='white')
            ax.tick_params(colors='white')
            plt.xticks(rotation=45)
            for spine in ax.spines.values():
                spine.set_color('rgba(255,255,255,0.1)')
            plt.tight_layout()
            st.pyplot(fig)
    
    # ═════════════════════════════════════════════════════════════
    # TAB 5: RECOMMENDATIONS
    # ═════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("#### 💡 AI-Powered Business Recommendations")
        
        # Calculate key insights
        top_age = df.groupby('Age')['Purchase'].mean().idxmax()
        top_age_val = df.groupby('Age')['Purchase'].mean().max()
        top_gender = df.groupby('Gender')['Purchase'].mean().idxmax()
        top_cat = df['Product_Category_1'].value_counts().index[0]
        top_revenue_cat = df.groupby('Product_Category_1')['Purchase'].sum().idxmax()
        
        st.markdown(f"""
        <div class="success-box">
            <h3>🎯 Key Insights Discovered</h3>
            <ul>
                <li><b>Top Spending Age Group:</b> {top_age} years (Avg: ${top_age_val:,.0f})</li>
                <li><b>Higher Spending Gender:</b> {top_gender}</li>
                <li><b>Most Popular Category:</b> Category {top_cat}</li>
                <li><b>Highest Revenue Category:</b> Category {top_revenue_cat}</li>
                <li><b>High-Value Customers Detected:</b> {len(anomalies):,} anomaly transactions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🚀 Strategic Recommendations")
        
        recs = [
            ("🎯 Targeted Marketing", f"Focus campaigns on the **{top_age}** age group - they have the highest average spend of ${top_age_val:,.0f}"),
            ("📦 Inventory Optimization", f"Stock more products from **Category {top_revenue_cat}** - it generates the most revenue"),
            ("🎁 Bundle Offers", "Create product bundles based on discovered association rules to increase cross-selling"),
            ("⭐ VIP Program", f"Launch a loyalty program for the {len(anomalies):,} identified high-value customers"),
            ("📱 Personalization", "Use cluster insights to personalize email campaigns and promotions"),
            ("🛒 Gender-Specific Promotions", f"Tailor product recommendations based on {top_gender} shopping preferences")
        ]
        
        cols = st.columns(2)
        for i, (title, desc) in enumerate(recs):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="insight-box">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Final summary
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(0, 245, 212, 0.15) 0%, rgba(155, 93, 229, 0.15) 100%); border: 1px solid rgba(0, 245, 212, 0.3); border-radius: 24px;">
            <h2 style="background: linear-gradient(135deg, #00f5d4 0%, #9b5de5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">🎉 Analysis Complete!</h2>
            <p style="color: rgba(255,255,255,0.7); font-size: 1.1rem;">Your Black Friday sales data has been analyzed successfully.</p>
            <p style="color: rgba(255,255,255,0.5);">Use these insights to optimize your retail strategy and boost revenue!</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # No data loaded - show welcome screen
    st.markdown("""
    <div class="main-header">
        <h1>🛍️ Black Friday Sales Analytics</h1>
        <p>Uncover hidden patterns in retail data with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.warning("⚠️ Dataset not found! Please ensure 'BlackFriday.csv' is in the same directory.")
    
    st.markdown("""
    ### 📋 Expected Dataset Format
    
    Your CSV file should contain the following columns:
    - **User_ID** - Unique customer identifier
    - **Product_ID** - Unique product identifier  
    - **Gender** - Customer gender (M/F)
    - **Age** - Age group (0-17, 18-25, 26-35, 36-45, 46-50, 51-55, 55+)
    - **Occupation** - Occupation code
    - **City_Category** - City category (A, B, C)
    - **Stay_In_Current_City_Years** - Years in current city
    - **Marital_Status** - Marital status (0/1)
    - **Product_Category_1, 2, 3** - Product categories
    - **Purchase** - Purchase amount in dollars
    """)
