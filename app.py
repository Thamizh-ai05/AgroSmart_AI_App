import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="AgroSmart AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API KEYS (move to .env in production)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGRO_API_KEY = os.environ.get("AGRO_API_KEY")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")  # add your key here

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM CSS — Professional Dark-Agricultural Theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --green-primary: #16a34a;
        --green-light: #22c55e;
        --green-dark: #14532d;
        --green-muted: #166534;
        --amber: #f59e0b;
        --amber-light: #fcd34d;
        --red-risk: #ef4444;
        --bg-primary: #0a0f0a;
        --bg-secondary: #111a11;
        --bg-card: #152015;
        --bg-card-hover: #1a2e1a;
        --border-subtle: #1f3320;
        --text-primary: #f0fdf4;
        --text-muted: #86efac;
        --text-faint: #4ade80;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    .stApp {
        background-color: var(--bg-primary);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1a0d 0%, #0a120a 100%);
        border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    /* ── Cards ── */
    .agro-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .agro-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--green-primary), var(--green-light), transparent);
    }
    .agro-card:hover {
        border-color: var(--green-muted);
        box-shadow: 0 0 20px rgba(22, 163, 74, 0.12);
    }

    /* ── Hero ── */
    .hero-banner {
        background: linear-gradient(135deg, #0d2010 0%, #0a1a0a 40%, #0d1a0f 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(22,163,74,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ade80, #22c55e, #86efac);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 1rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* ── Metric Cards ── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: all 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: var(--green-light);
        box-shadow: 0 4px 20px rgba(34,197,94,0.1);
    }
    .metric-icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--green-light);
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
    }
    .metric-unit {
        font-size: 0.8rem;
        color: #6b7280;
    }

    /* ── Crop Result ── */
    .crop-result-card {
        background: linear-gradient(135deg, #0d2010, #112211);
        border: 2px solid var(--green-primary);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 40px rgba(22,163,74,0.15), inset 0 0 40px rgba(22,163,74,0.03);
    }
    .crop-result-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% -20%, rgba(34,197,94,0.08) 0%, transparent 60%);
    }
    .crop-name {
        font-size: 3rem;
        font-weight: 800;
        color: #4ade80;
        letter-spacing: -1px;
        text-transform: uppercase;
    }
    .crop-confidence {
        display: inline-block;
        background: rgba(22,163,74,0.2);
        border: 1px solid var(--green-primary);
        border-radius: 999px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        color: var(--green-light);
        margin-top: 0.5rem;
    }

    /* ── Rank Badge ── */
    .rank-badge {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        transition: all 0.2s;
    }
    .rank-badge:hover { border-color: var(--green-muted); }
    .rank-num {
        font-size: 1.4rem;
        font-weight: 800;
        color: var(--green-light);
        width: 2rem;
        text-align: center;
    }
    .rank-name { font-size: 1rem; font-weight: 600; }
    .rank-bar-bg {
        flex: 1;
        background: rgba(22,163,74,0.1);
        border-radius: 999px;
        height: 6px;
        overflow: hidden;
    }
    .rank-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--green-primary), var(--green-light));
    }
    .rank-pct {
        font-size: 0.8rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        min-width: 3rem;
        text-align: right;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--green-light);
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-subtle);
    }

    /* ── Tags ── */
    .tag-green {
        display: inline-block;
        background: rgba(22,163,74,0.15);
        border: 1px solid var(--green-muted);
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
        color: var(--green-light);
        margin: 0.15rem;
    }
    .tag-amber {
        display: inline-block;
        background: rgba(245,158,11,0.12);
        border: 1px solid #92400e;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
        color: var(--amber-light);
        margin: 0.15rem;
    }
    .tag-red {
        display: inline-block;
        background: rgba(239,68,68,0.12);
        border: 1px solid #991b1b;
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.78rem;
        color: #fca5a5;
        margin: 0.15rem;
    }

    /* ── Alert Boxes ── */
    .alert-success {
        background: rgba(22,163,74,0.08);
        border-left: 3px solid var(--green-primary);
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #bbf7d0;
    }
    .alert-warning {
        background: rgba(245,158,11,0.08);
        border-left: 3px solid var(--amber);
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #fef08a;
    }
    .alert-danger {
        background: rgba(239,68,68,0.08);
        border-left: 3px solid var(--red-risk);
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #fecaca;
    }

    /* ── Divider ── */
    .agro-divider {
        border: none;
        border-top: 1px solid var(--border-subtle);
        margin: 1.5rem 0;
    }

    /* ── Streamlit overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--green-primary), var(--green-dark));
        color: white;
        border: none;
        border-radius: 10px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
        box-shadow: 0 2px 8px rgba(22,163,74,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--green-light), var(--green-primary));
        box-shadow: 0 4px 16px rgba(34,197,94,0.4);
        transform: translateY(-1px);
    }

    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input,
    .stSlider > div {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-subtle) !important;
        border-radius: 8px !important;
        font-family: 'Outfit', sans-serif !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 12px;
        gap: 0.3rem;
        padding: 0.3rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-muted);
        border-radius: 8px;
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: var(--green-dark) !important;
        color: var(--green-light) !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: var(--green-light) !important;
    }

    .stProgress .st-bo { background-color: var(--green-primary) !important; }
    .stProgress .st-b8 { background-color: var(--bg-card) !important; }

    .footer-bar {
        text-align: center;
        padding: 1.5rem;
        color: #4b5563;
        font-size: 0.8rem;
        border-top: 1px solid var(--border-subtle);
        margin-top: 2rem;
    }
    .footer-bar span { color: var(--green-muted); }
    
    /* Hide default Streamlit header */
    #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE INIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
defaults = {
    "name": None,
    "city": "Chennai",
    "manual_soil": False,
    "soil_N": 75.0,
    "soil_P": 40.0,
    "soil_K": 40.0,
    "soil_ph": 6.5,
    "prediction_history": [],
    "model_accuracy": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPREHENSIVE CROP DATABASE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CROP_DB = {
    "rice": {
        "emoji": "🌾", "season": "Kharif",
        "temp_range": (20, 35), "rain_range": (100, 200), "ph_range": (5.5, 7.0),
        "cost_per_acre": 30000, "yield_quintals": 25, "price_per_quintal": 2200,
        "duration_days": 120, "water_need": "High",
        "fertilizer": {"N": 100, "P": 50, "K": 50},
        "best_soil": "Clay, Loamy",
        "ideal_regions": ["Tamil Nadu", "Punjab", "West Bengal", "Andhra Pradesh"],
        "tips": ["Transplant at 25-day-old seedlings", "Maintain 5cm water level during tillering", "Drain field 10 days before harvest"],
        "pests": ["Brown plant hopper", "Stem borer", "Leaf folder"],
    },
    "wheat": {
        "emoji": "🌿", "season": "Rabi",
        "temp_range": (10, 25), "rain_range": (40, 100), "ph_range": (6.0, 7.5),
        "cost_per_acre": 25000, "yield_quintals": 20, "price_per_quintal": 2100,
        "duration_days": 120, "water_need": "Medium",
        "fertilizer": {"N": 120, "P": 60, "K": 40},
        "best_soil": "Loamy, Clay loam",
        "ideal_regions": ["Punjab", "Haryana", "UP", "MP"],
        "tips": ["Sow at 20-25°C", "Apply first irrigation at Crown Root Stage", "Timely harvest prevents shattering"],
        "pests": ["Yellow rust", "Aphids", "Termites"],
    },
    "maize": {
        "emoji": "🌽", "season": "Kharif",
        "temp_range": (18, 35), "rain_range": (60, 110), "ph_range": (5.8, 7.0),
        "cost_per_acre": 18000, "yield_quintals": 30, "price_per_quintal": 1700,
        "duration_days": 90, "water_need": "Medium",
        "fertilizer": {"N": 120, "P": 60, "K": 40},
        "best_soil": "Sandy loam, Silt loam",
        "ideal_regions": ["Karnataka", "AP", "Rajasthan", "Bihar"],
        "tips": ["Plant in rows of 60cm x 25cm spacing", "Apply Zinc sulphate for better yield", "Detasseling boosts seed production"],
        "pests": ["Fall army worm", "Stem borer", "Thrips"],
    },
    "cotton": {
        "emoji": "🌸", "season": "Kharif",
        "temp_range": (21, 35), "rain_range": (50, 100), "ph_range": (5.8, 8.0),
        "cost_per_acre": 35000, "yield_quintals": 8, "price_per_quintal": 6500,
        "duration_days": 180, "water_need": "Medium",
        "fertilizer": {"N": 100, "P": 50, "K": 50},
        "best_soil": "Black cotton, Alluvial",
        "ideal_regions": ["Gujarat", "Maharashtra", "Telangana", "Punjab"],
        "tips": ["Maintain plant population of 11,000/acre", "Topping at 9th week promotes branching", "Drip irrigation improves yield 20-30%"],
        "pests": ["Pink bollworm", "Whitefly", "Thrips"],
    },
    "sugarcane": {
        "emoji": "🌱", "season": "Annual",
        "temp_range": (20, 40), "rain_range": (100, 175), "ph_range": (6.0, 7.5),
        "cost_per_acre": 45000, "yield_quintals": 400, "price_per_quintal": 320,
        "duration_days": 365, "water_need": "Very High",
        "fertilizer": {"N": 250, "P": 80, "K": 120},
        "best_soil": "Loamy, Sandy loam",
        "ideal_regions": ["UP", "Maharashtra", "Karnataka", "Tamil Nadu"],
        "tips": ["Use disease-free setts for planting", "Trash mulching saves irrigation water", "Ratoon crop reduces cost by 40%"],
        "pests": ["Top shoot borer", "Pyrilla", "White grubs"],
    },
    "coffee": {
        "emoji": "☕", "season": "Perennial",
        "temp_range": (15, 28), "rain_range": (150, 250), "ph_range": (5.5, 6.5),
        "cost_per_acre": 60000, "yield_quintals": 6, "price_per_quintal": 25000,
        "duration_days": 365, "water_need": "High",
        "fertilizer": {"N": 100, "P": 45, "K": 75},
        "best_soil": "Red loamy, Laterite",
        "ideal_regions": ["Karnataka", "Kerala", "Tamil Nadu"],
        "tips": ["Intercrop with pepper or cardamom for extra income", "Shade management critical for quality", "Pulping within 12 hours of harvest"],
        "pests": ["Coffee berry borer", "White stem borer", "Leaf rust"],
    },
    "mango": {
        "emoji": "🥭", "season": "Summer",
        "temp_range": (24, 38), "rain_range": (50, 125), "ph_range": (5.5, 7.5),
        "cost_per_acre": 40000, "yield_quintals": 50, "price_per_quintal": 3500,
        "duration_days": 365, "water_need": "Medium",
        "fertilizer": {"N": 80, "P": 40, "K": 80},
        "best_soil": "Deep alluvial, Sandy loam",
        "ideal_regions": ["UP", "AP", "Maharashtra", "Bihar", "Tamil Nadu"],
        "tips": ["Paclobutrazol spray induces off-year flowering", "Mango malformation manageable with NAA spray", "Harvest at maturity — premature harvest affects quality"],
        "pests": ["Mango hopper", "Fruit fly", "Powdery mildew"],
    },
    "banana": {
        "emoji": "🍌", "season": "Annual",
        "temp_range": (20, 35), "rain_range": (100, 200), "ph_range": (5.5, 7.0),
        "cost_per_acre": 50000, "yield_quintals": 200, "price_per_quintal": 900,
        "duration_days": 365, "water_need": "High",
        "fertilizer": {"N": 200, "P": 60, "K": 300},
        "best_soil": "Rich loamy, Clay loam",
        "ideal_regions": ["Tamil Nadu", "AP", "Karnataka", "Maharashtra"],
        "tips": ["Tissue culture plants give uniform and disease-free crop", "Prop bunches to prevent toppling", "Harvest when fingers turn light green"],
        "pests": ["Banana weevil", "Nematodes", "Sigatoka leaf spot"],
    },
    "tomato": {
        "emoji": "🍅", "season": "Rabi/Summer",
        "temp_range": (18, 30), "rain_range": (60, 100), "ph_range": (5.5, 7.0),
        "cost_per_acre": 55000, "yield_quintals": 200, "price_per_quintal": 1200,
        "duration_days": 90, "water_need": "Medium",
        "fertilizer": {"N": 120, "P": 80, "K": 100},
        "best_soil": "Sandy loam, Red loam",
        "ideal_regions": ["Maharashtra", "Karnataka", "AP", "Tamil Nadu"],
        "tips": ["Staking prevents fruit rot and improves quality", "Mulching reduces weed and conserves moisture", "Foliar spray of Boron at flowering boosts yield"],
        "pests": ["Fruit borer", "Whitefly", "Early blight"],
    },
    "chickpea": {
        "emoji": "🫘", "season": "Rabi",
        "temp_range": (10, 25), "rain_range": (60, 90), "ph_range": (6.0, 8.0),
        "cost_per_acre": 15000, "yield_quintals": 10, "price_per_quintal": 5000,
        "duration_days": 110, "water_need": "Low",
        "fertilizer": {"N": 30, "P": 60, "K": 40},
        "best_soil": "Sandy loam, Medium black",
        "ideal_regions": ["MP", "Rajasthan", "UP", "Maharashtra"],
        "tips": ["Rhizobium seed treatment improves nitrogen fixation", "Avoid excess N fertilizer — reduces yield", "Irrigate at pre-flowering and pod-filling stages"],
        "pests": ["Pod borer", "Aphids", "Fusarium wilt"],
    },
    "lentil": {
        "emoji": "🌰", "season": "Rabi",
        "temp_range": (10, 25), "rain_range": (30, 70), "ph_range": (5.8, 7.5),
        "cost_per_acre": 12000, "yield_quintals": 8, "price_per_quintal": 5500,
        "duration_days": 100, "water_need": "Low",
        "fertilizer": {"N": 20, "P": 50, "K": 30},
        "best_soil": "Sandy loam, Alluvial",
        "ideal_regions": ["MP", "UP", "Bihar", "Rajasthan"],
        "tips": ["Seed inoculation with Rhizobium reduces N cost", "One pre-sowing irrigation is critical", "Harvest when 70% pods turn brown"],
        "pests": ["Aphids", "Rust", "Powdery mildew"],
    },
    "muskmelon": {
        "emoji": "🍈", "season": "Summer",
        "temp_range": (25, 38), "rain_range": (25, 60), "ph_range": (6.0, 7.5),
        "cost_per_acre": 20000, "yield_quintals": 90, "price_per_quintal": 1500,
        "duration_days": 80, "water_need": "Low-Medium",
        "fertilizer": {"N": 60, "P": 40, "K": 60},
        "best_soil": "Sandy loam, Light alluvial",
        "ideal_regions": ["Rajasthan", "UP", "AP", "Tamil Nadu"],
        "tips": ["Train vines on trellis for better air circulation", "Pollination in early morning gives best fruit set", "Reduce irrigation 10 days before harvest improves sweetness"],
        "pests": ["Red pumpkin beetle", "Fruit fly", "Downy mildew"],
    },
    "watermelon": {
        "emoji": "🍉", "season": "Summer",
        "temp_range": (24, 38), "rain_range": (40, 75), "ph_range": (5.5, 7.0),
        "cost_per_acre": 18000, "yield_quintals": 120, "price_per_quintal": 900,
        "duration_days": 75, "water_need": "Medium",
        "fertilizer": {"N": 80, "P": 40, "K": 60},
        "best_soil": "Sandy loam",
        "ideal_regions": ["AP", "Karnataka", "UP", "Rajasthan"],
        "tips": ["Leave 2-3 fruits per vine for large size", "Drip fertigation maximises yield", "Check ripeness by tapping — hollow sound indicates ripe"],
        "pests": ["Fruit fly", "Aphids", "Powdery mildew"],
    },
    "onion": {
        "emoji": "🧅", "season": "Rabi",
        "temp_range": (13, 30), "rain_range": (35, 75), "ph_range": (5.8, 7.0),
        "cost_per_acre": 40000, "yield_quintals": 100, "price_per_quintal": 1500,
        "duration_days": 120, "water_need": "Medium",
        "fertilizer": {"N": 100, "P": 50, "K": 60},
        "best_soil": "Sandy loam, Medium black",
        "ideal_regions": ["Maharashtra", "Karnataka", "MP", "Gujarat"],
        "tips": ["Raised beds improve drainage and reduce rots", "Irrigation critical during bulb formation", "Curing after harvest reduces storage losses"],
        "pests": ["Thrips", "Purple blotch", "Damping off"],
    },
    "groundnut": {
        "emoji": "🥜", "season": "Kharif",
        "temp_range": (20, 35), "rain_range": (50, 125), "ph_range": (5.5, 7.0),
        "cost_per_acre": 22000, "yield_quintals": 12, "price_per_quintal": 5500,
        "duration_days": 120, "water_need": "Medium",
        "fertilizer": {"N": 25, "P": 50, "K": 50},
        "best_soil": "Sandy loam, Red sandy",
        "ideal_regions": ["Gujarat", "AP", "Tamil Nadu", "Karnataka"],
        "tips": ["Gypsum at pegging stage improves pod filling", "Avoid waterlogging — causes collar rot", "Harvest when leaves turn yellow and pegs darken"],
        "pests": ["Leaf miner", "Groundnut bud necrosis", "Tikka disease"],
    },
    "soybean": {
        "emoji": "🫘", "season": "Kharif",
        "temp_range": (20, 32), "rain_range": (60, 100), "ph_range": (6.0, 7.5),
        "cost_per_acre": 20000, "yield_quintals": 12, "price_per_quintal": 4500,
        "duration_days": 110, "water_need": "Medium",
        "fertilizer": {"N": 30, "P": 60, "K": 40},
        "best_soil": "Well-drained loam, Black cotton",
        "ideal_regions": ["MP", "Maharashtra", "Rajasthan"],
        "tips": ["Rhizobium + PSB inoculation boosts yield by 15%", "Avoid excess irrigation to prevent lodging", "Harvest at 92-94% pod maturity"],
        "pests": ["Stem fly", "Girdle beetle", "Yellow mosaic virus"],
    },
    "papaya": {
        "emoji": "🍑", "season": "Annual",
        "temp_range": (22, 35), "rain_range": (100, 150), "ph_range": (6.0, 7.0),
        "cost_per_acre": 35000, "yield_quintals": 300, "price_per_quintal": 1200,
        "duration_days": 365, "water_need": "High",
        "fertilizer": {"N": 200, "P": 100, "K": 200},
        "best_soil": "Sandy loam, Alluvial",
        "ideal_regions": ["AP", "Tamil Nadu", "Maharashtra", "Gujarat"],
        "tips": ["Plant gynoecious hybrids for all-female population", "Papaya ring spot virus — use certified virus-free seedlings", "Harvest when two to three strips turn yellow"],
        "pests": ["Papaya ring spot virus", "Mites", "Fruit fly"],
    },
    "coconut": {
        "emoji": "🥥", "season": "Perennial",
        "temp_range": (27, 35), "rain_range": (100, 250), "ph_range": (5.5, 7.5),
        "cost_per_acre": 80000, "yield_quintals": 75, "price_per_quintal": 2000,
        "duration_days": 365, "water_need": "High",
        "fertilizer": {"N": 500, "P": 320, "K": 1200},
        "best_soil": "Sandy loam, Laterite, Alluvial",
        "ideal_regions": ["Kerala", "Tamil Nadu", "Karnataka", "AP"],
        "tips": ["Drip irrigation saves 30-40% water", "Intercrop with cocoa, arecanut, or banana", "Bud rot: Remove infected crown and apply Bordeaux paste"],
        "pests": ["Rhinoceros beetle", "Red palm weevil", "Bud rot"],
    },
    "turmeric": {
        "emoji": "🌿", "season": "Kharif",
        "temp_range": (20, 30), "rain_range": (100, 200), "ph_range": (5.5, 7.0),
        "cost_per_acre": 60000, "yield_quintals": 80, "price_per_quintal": 7500,
        "duration_days": 240, "water_range": "High",
        "fertilizer": {"N": 60, "P": 50, "K": 120},
        "water_need": "High",
        "best_soil": "Sandy loam, Clay loam",
        "ideal_regions": ["Andhra Pradesh", "Tamil Nadu", "Odisha", "Maharashtra"],
        "tips": ["Mulching with green leaves reduces weeding cost", "Curing determines colour, flavour, and price", "Mother rhizomes give better sprouts than finger rhizomes"],
        "pests": ["Rhizome rot", "Leaf blotch", "Shoot borer"],
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYNTHETIC TRAINING DATA GENERATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data
def generate_training_data(n_samples=3000):
    """
    Generate realistic synthetic training data based on agronomic domain knowledge.
    Each crop has specific climate/soil requirements with gaussian noise.
    """
    np.random.seed(42)
    rows = []

    crop_params = {
        "rice":       {"N":(80,15), "P":(45,10), "K":(45,10), "temp":(27,4),  "humidity":(80,8),  "ph":(6.0,0.4), "rain":(180,30)},
        "wheat":      {"N":(90,15), "P":(50,10), "K":(40,10), "temp":(18,4),  "humidity":(65,8),  "ph":(6.5,0.4), "rain":(70,20)},
        "maize":      {"N":(80,15), "P":(48,10), "K":(50,10), "temp":(25,5),  "humidity":(65,10), "ph":(6.2,0.5), "rain":(80,20)},
        "cotton":     {"N":(90,15), "P":(46,10), "K":(48,10), "temp":(28,5),  "humidity":(60,10), "ph":(6.5,0.7), "rain":(75,20)},
        "sugarcane":  {"N":(180,20),"P":(60,10), "K":(100,15),"temp":(30,5),  "humidity":(75,8),  "ph":(6.3,0.5), "rain":(150,30)},
        "coffee":     {"N":(90,15), "P":(40,8),  "K":(65,12), "temp":(22,4),  "humidity":(75,8),  "ph":(6.0,0.4), "rain":(200,30)},
        "mango":      {"N":(70,15), "P":(35,8),  "K":(70,12), "temp":(30,5),  "humidity":(55,10), "ph":(6.0,0.5), "rain":(80,20)},
        "banana":     {"N":(140,20),"P":(55,10), "K":(200,25),"temp":(28,4),  "humidity":(75,8),  "ph":(6.2,0.4), "rain":(160,30)},
        "tomato":     {"N":(100,15),"P":(70,12), "K":(90,15), "temp":(24,4),  "humidity":(70,8),  "ph":(6.0,0.4), "rain":(80,20)},
        "chickpea":   {"N":(25,8),  "P":(55,10), "K":(38,8),  "temp":(18,4),  "humidity":(55,8),  "ph":(6.8,0.5), "rain":(75,20)},
        "lentil":     {"N":(18,6),  "P":(45,8),  "K":(28,6),  "temp":(18,4),  "humidity":(50,8),  "ph":(6.5,0.5), "rain":(50,15)},
        "muskmelon":  {"N":(55,10), "P":(35,8),  "K":(55,10), "temp":(32,5),  "humidity":(70,10), "ph":(6.5,0.5), "rain":(40,15)},
        "watermelon": {"N":(65,10), "P":(35,8),  "K":(55,10), "temp":(30,5),  "humidity":(72,10), "ph":(6.0,0.5), "rain":(55,15)},
        "onion":      {"N":(85,12), "P":(45,10), "K":(55,10), "temp":(22,5),  "humidity":(60,8),  "ph":(6.0,0.5), "rain":(55,15)},
        "groundnut":  {"N":(22,6),  "P":(45,8),  "K":(45,8),  "temp":(27,5),  "humidity":(65,8),  "ph":(6.0,0.5), "rain":(85,20)},
        "soybean":    {"N":(28,8),  "P":(55,10), "K":(38,8),  "temp":(26,4),  "humidity":(68,8),  "ph":(6.5,0.5), "rain":(80,20)},
        "papaya":     {"N":(160,20),"P":(80,15), "K":(160,20),"temp":(28,4),  "humidity":(75,8),  "ph":(6.5,0.4), "rain":(120,25)},
        "coconut":    {"N":(300,30),"P":(200,25),"K":(600,50),"temp":(30,4),  "humidity":(80,8),  "ph":(6.0,0.5), "rain":(175,30)},
        "turmeric":   {"N":(55,10), "P":(45,8),  "K":(110,15),"temp":(25,4),  "humidity":(78,8),  "ph":(6.0,0.4), "rain":(150,30)},
    }

    per_crop = n_samples // len(crop_params)
    for crop, params in crop_params.items():
        for _ in range(per_crop):
            row = {
                "N":        np.clip(np.random.normal(params["N"][0],    params["N"][1]),    0, 300),
                "P":        np.clip(np.random.normal(params["P"][0],    params["P"][1]),    0, 250),
                "K":        np.clip(np.random.normal(params["K"][0],    params["K"][1]),    0, 700),
                "temperature": np.clip(np.random.normal(params["temp"][0],  params["temp"][1]),  5, 45),
                "humidity": np.clip(np.random.normal(params["humidity"][0], params["humidity"][1]), 20, 100),
                "ph":       np.clip(np.random.normal(params["ph"][0],   params["ph"][1]),   4, 9),
                "rainfall": np.clip(np.random.normal(params["rain"][0],  params["rain"][1]),  10, 300),
                "label":    crop
            }
            rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource
def train_model():
    """
    Train an ensemble of RandomForest + GradientBoosting for robust predictions.
    Returns: (ensemble_predict_fn, classes, accuracy, feature_importances)
    """
    df = generate_training_data(3000)
    
    # Try loading real CSV if available
    if os.path.exists("crop_data.csv"):
        try:
            real_df = pd.read_csv("crop_data.csv")
            # Rename common column variants
            col_map = {"label": "label", "crop": "label"}
            real_df = real_df.rename(columns={c: col_map.get(c, c) for c in real_df.columns})
            if "label" in real_df.columns:
                df = pd.concat([df, real_df], ignore_index=True)
        except Exception:
            pass

    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    acc_gb = accuracy_score(y_test, gb.predict(X_test))

    # Weighted ensemble
    def ensemble_predict_proba(X_df):
        p_rf = rf.predict_proba(X_df)
        p_gb = gb.predict_proba(X_df)
        return 0.55 * p_rf + 0.45 * p_gb

    classes = rf.classes_
    acc = round((acc_rf * 0.55 + acc_gb * 0.45) * 100, 1)
    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return ensemble_predict_proba, classes, acc, feat_imp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEO & WEATHER APIs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data(ttl=3600)
def geocode(city: str):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        r = requests.get(url, params={"q": city, "format": "json"}, 
                         headers={"User-Agent": "agrosmart-ai-v2"}, timeout=6)
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", city)
    except Exception:
        pass
    return 13.08, 80.27, "Chennai, Tamil Nadu"


@st.cache_data(ttl=600)
def fetch_weather(lat: float, lon: float):
    """
    Fetch weather from AgroMonitoring API with proper error handling.
    Returns dict with all weather fields or None on failure.
    """
    try:
        url = f"http://api.agromonitoring.com/agro/1.0/weather"
        r = requests.get(url, params={"lat": lat, "lon": lon, "appid": AGRO_API_KEY}, timeout=8)
        if r.status_code == 200:
            d = r.json()
            return {
                "temp":     round(d["main"]["temp"] - 273.15, 1),
                "feels_like": round(d["main"].get("feels_like", d["main"]["temp"]) - 273.15, 1),
                "humidity": d["main"]["humidity"],
                "pressure": d["main"].get("pressure", 1013),
                "rainfall": round(d.get("rain", {}).get("1h", 0) * 30, 1),  # extrapolate monthly
                "wind_speed": round(d.get("wind", {}).get("speed", 0) * 3.6, 1),  # m/s to km/h
                "description": d["weather"][0]["description"].title() if d.get("weather") else "Clear",
                "icon": d["weather"][0]["icon"] if d.get("weather") else "01d",
                "source": "AgroMonitoring API"
            }
    except Exception:
        pass

    # Fallback: try OpenWeatherMap if key exists
    if OPENWEATHER_API_KEY:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather"
            r = requests.get(url, params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}, timeout=8)
            if r.status_code == 200:
                d = r.json()
                return {
                    "temp":       round(d["main"]["temp"] - 273.15, 1),
                    "feels_like": round(d["main"].get("feels_like", d["main"]["temp"]) - 273.15, 1),
                    "humidity":   d["main"]["humidity"],
                    "pressure":   d["main"].get("pressure", 1013),
                    "rainfall":   round(d.get("rain", {}).get("1h", 0) * 30, 1),
                    "wind_speed": round(d.get("wind", {}).get("speed", 0) * 3.6, 1),
                    "description": d["weather"][0]["description"].title() if d.get("weather") else "Clear",
                    "icon": d["weather"][0]["icon"] if d.get("weather") else "01d",
                    "source": "OpenWeatherMap API"
                }
        except Exception:
            pass

    # Fallback: climatological defaults based on lat/lon zone
    if lat < 15:
        temp, hum, rain = 32.0, 78, 180  # Tropical south
    elif lat < 22:
        temp, hum, rain = 28.0, 68, 90   # Central India
    else:
        temp, hum, rain = 22.0, 55, 60   # North India

    return {
        "temp": temp, "feels_like": temp - 1.5, "humidity": hum,
        "pressure": 1013, "rainfall": rain, "wind_speed": 12,
        "description": "Partly Cloudy (Estimated)",
        "icon": "02d", "source": "Climatological Estimate"
    }


def infer_soil_from_region(city: str, temp: float, rain: float, humidity: float) -> dict:
    """
    Region + climate-aware soil parameter estimation.
    Based on ICAR soil classification data for Indian regions.
    """
    city_l = city.lower()

    # Base defaults
    N, P, K, ph = 75.0, 40.0, 40.0, 6.5

    # Tamil Nadu districts
    if any(x in city_l for x in ["chennai", "vellore", "thiruvannamalai", "villupuram"]):
        N, P, K, ph = 68, 38, 42, 6.8  # Red laterite
    elif any(x in city_l for x in ["coimbatore", "erode", "tiruppur", "salem"]):
        N, P, K, ph = 72, 42, 55, 6.4  # Red loam
    elif any(x in city_l for x in ["madurai", "dindigul", "theni"]):
        N, P, K, ph = 78, 44, 38, 6.6  # Black cotton
    elif any(x in city_l for x in ["thanjavur", "nagapattinam", "tiruvarur", "kumbakonam"]):
        N, P, K, ph = 88, 46, 48, 6.2  # Alluvial (Kaveri delta)
    elif any(x in city_l for x in ["nilgiris", "ooty", "kodaikanal"]):
        N, P, K, ph = 82, 38, 44, 5.8  # Mountain laterite
    # Andhra / Telangana
    elif any(x in city_l for x in ["hyderabad", "warangal", "karimnagar"]):
        N, P, K, ph = 75, 40, 35, 7.2  # Red / black
    elif any(x in city_l for x in ["visakhapatnam", "vizag", "rajahmundry"]):
        N, P, K, ph = 80, 42, 45, 6.5  # Alluvial
    # Karnataka
    elif any(x in city_l for x in ["bengaluru", "bangalore", "mysuru", "mysore"]):
        N, P, K, ph = 70, 36, 42, 6.0  # Red laterite
    elif any(x in city_l for x in ["hubli", "dharwad", "belgaum", "belagavi"]):
        N, P, K, ph = 76, 40, 38, 7.0  # Black cotton
    # Kerala
    elif any(x in city_l for x in ["thiruvananthapuram", "kochi", "kozhikode", "thrissur"]):
        N, P, K, ph = 65, 32, 38, 5.8  # Laterite
    # Maharashtra
    elif any(x in city_l for x in ["pune", "nashik", "solapur", "aurangabad"]):
        N, P, K, ph = 82, 45, 42, 7.2  # Black vertisol
    elif any(x in city_l for x in ["mumbai", "thane", "nagpur"]):
        N, P, K, ph = 78, 42, 40, 6.8
    # North India
    elif any(x in city_l for x in ["delhi", "new delhi", "gurgaon", "noida"]):
        N, P, K, ph = 85, 48, 42, 7.5  # Alluvial IGP
    elif any(x in city_l for x in ["lucknow", "kanpur", "varanasi", "allahabad"]):
        N, P, K, ph = 88, 50, 45, 7.4  # Deep alluvial
    elif any(x in city_l for x in ["patna", "gaya", "muzaffarpur"]):
        N, P, K, ph = 90, 52, 48, 6.8  # Alluvial Bihar
    elif any(x in city_l for x in ["amritsar", "ludhiana", "jalandhar", "chandigarh"]):
        N, P, K, ph = 92, 55, 50, 7.6  # Punjab alluvial

    # Climate modifiers on top of region base
    if rain > 150:
        N  = max(N  - 8,  20)
        ph = max(ph - 0.3, 5.0)  # leaching under heavy rain
    if rain < 40:
        K  = min(K  + 5,  120)
        ph = min(ph + 0.3,  8.5)
    if temp > 35:
        N  = min(N  + 6,  200)
    if humidity > 75:
        P  = max(P  - 3,  10)

    return {"N": round(N, 1), "P": round(P, 1), "K": round(K, 1), "ph": round(ph, 2)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ECONOMIC ANALYSIS ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_economics(crop_name: str, acres: float = 1.0) -> dict:
    info = CROP_DB.get(crop_name)
    if not info:
        return {}

    cost         = info["cost_per_acre"] * acres
    yield_q      = info["yield_quintals"] * acres
    yield_kg     = yield_q * 100
    price_per_q  = info["price_per_quintal"]
    revenue      = yield_q * price_per_q
    net_profit   = revenue - cost
    roi          = (net_profit / cost) * 100 if cost else 0
    duration     = info["duration_days"]
    profit_day   = net_profit / duration

    # Breakeven yield
    breakeven_q = cost / price_per_q

    return {
        "cost":         cost,
        "yield_q":      yield_q,
        "yield_kg":     yield_kg,
        "price_per_q":  price_per_q,
        "revenue":      revenue,
        "net_profit":   net_profit,
        "roi":          round(roi, 1),
        "duration":     duration,
        "profit_day":   round(profit_day, 1),
        "breakeven_q":  round(breakeven_q, 1),
        "price_per_kg": round(price_per_q / 100, 1),
    }


def get_suitability_score(crop_name: str, temp: float, rain: float, ph: float,
                           humidity: float) -> tuple[int, list[str], list[str]]:
    """Score crop suitability 0-100 with reasons and warnings."""
    info = CROP_DB.get(crop_name)
    if not info:
        return 50, [], []

    score = 100
    reasons = []
    warnings = []

    t_min, t_max = info["temp_range"]
    r_min, r_max = info["rain_range"]
    p_min, p_max = info["ph_range"]

    # Temperature
    if t_min <= temp <= t_max:
        reasons.append(f"✅ Temperature {temp}°C is ideal ({t_min}–{t_max}°C)")
    elif temp < t_min - 5 or temp > t_max + 5:
        score -= 25
        warnings.append(f"🌡 Temp {temp}°C is far outside ideal range ({t_min}–{t_max}°C)")
    else:
        score -= 10
        warnings.append(f"⚠ Temperature slightly outside ideal range ({t_min}–{t_max}°C)")

    # Rainfall
    if r_min <= rain <= r_max:
        reasons.append(f"✅ Rainfall {rain}mm/month suits this crop ({r_min}–{r_max}mm)")
    elif rain < r_min * 0.5:
        score -= 20
        warnings.append(f"⚠ Low rainfall — consider irrigation")
    elif rain > r_max * 1.5:
        score -= 15
        warnings.append(f"⚠ Excess rainfall — drainage management needed")
    else:
        score -= 8

    # pH
    if p_min <= ph <= p_max:
        reasons.append(f"✅ Soil pH {ph} is in optimal range ({p_min}–{p_max})")
    else:
        gap = min(abs(ph - p_min), abs(ph - p_max))
        score -= min(int(gap * 12), 20)
        if ph < p_min:
            warnings.append(f"⚠ Acidic soil (pH {ph}) — apply lime to raise to {p_min}")
        else:
            warnings.append(f"⚠ Alkaline soil (pH {ph}) — apply gypsum / sulphur")

    # Humidity bonus/penalty
    if humidity > 85:
        score -= 5
        warnings.append("⚠ High humidity may increase fungal disease pressure")
    elif humidity > 60:
        reasons.append("✅ Humidity level supports good growth")

    return max(0, min(100, score)), reasons, warnings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NAME ONBOARDING SCREEN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if st.session_state.name is None:
    st.markdown("""
    <div style='display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:60vh;'>
        <div style='text-align:center; max-width:480px;'>
            <div style='font-size:4rem; margin-bottom:1rem;'>🌾</div>
            <h1 style='font-size:2.8rem; font-weight:800;
                       background:linear-gradient(135deg,#4ade80,#22c55e);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                       background-clip:text; margin:0;'>
                AgroSmart AI
            </h1>
            <p style='color:#86efac; margin:0.5rem 0 2rem; font-size:1.05rem;'>
                Intelligent Crop Advisory · Real-time Weather · ML-Powered
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        nm = st.text_input("", placeholder="Enter your name to get started...", label_visibility="collapsed")
        if st.button("🌱  Let's Begin →", use_container_width=True):
            if nm.strip():
                st.session_state.name = nm.strip()
                st.rerun()
            else:
                st.error("Please enter your name")
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown(f"""
    <div style='padding:1rem; background:rgba(22,163,74,0.08);
                border:1px solid #1f3320; border-radius:12px; margin-bottom:1rem;'>
        <div style='font-size:1.5rem;'>👤</div>
        <div style='font-size:1.1rem; font-weight:600; color:#4ade80;'>
            {st.session_state.name}
        </div>
        <div style='font-size:0.78rem; color:#6b7280; margin-top:0.2rem;'>
            AgroSmart Member
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📍 Location")
    new_city = st.text_input("City / Region", value=st.session_state.city)
    if st.button("🔄 Update Location", use_container_width=True):
        if new_city.strip():
            st.session_state.city = new_city.strip()
            st.rerun()

    st.markdown("---")
    st.markdown("### 🌱 Farm Size")
    acres = st.number_input("Area (acres)", min_value=0.1, max_value=1000.0, value=1.0, step=0.5)

    st.markdown("---")
    st.markdown("### 🧪 Soil Input Mode")
    manual_mode = st.toggle("Enter soil values manually", value=st.session_state.manual_soil)
    st.session_state.manual_soil = manual_mode

    if manual_mode:
        st.markdown("**Custom Soil Parameters:**")
        st.session_state.soil_N  = st.slider("N — Nitrogen (kg/ha)",  0.0, 200.0, st.session_state.soil_N,  step=1.0)
        st.session_state.soil_P  = st.slider("P — Phosphorus (kg/ha)", 0.0, 200.0, st.session_state.soil_P, step=1.0)
        st.session_state.soil_K  = st.slider("K — Potassium (kg/ha)", 0.0, 700.0, st.session_state.soil_K,  step=1.0)
        st.session_state.soil_ph = st.slider("pH",                     4.0,   9.0, st.session_state.soil_ph, step=0.1)

    st.markdown("---")

    # Model accuracy badge
    if st.session_state.model_accuracy:
        st.markdown(f"""
        <div style='background:rgba(22,163,74,0.08); border:1px solid #1f3320;
                    border-radius:10px; padding:0.8rem; text-align:center;'>
            <div style='font-size:0.75rem; color:#6b7280; text-transform:uppercase;
                        letter-spacing:0.05em;'>Model Accuracy</div>
            <div style='font-size:1.8rem; font-weight:800; color:#4ade80;
                        font-family:"JetBrains Mono",monospace;'>
                {st.session_state.model_accuracy}%
            </div>
            <div style='font-size:0.72rem; color:#4b5563;'>Random Forest + GradBoost</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.prediction_history:
        st.markdown("### 📋 History")
        for i, h in enumerate(reversed(st.session_state.prediction_history[-5:])):
            st.markdown(f"<div class='tag-green'>{h['city']} → {h['crop']}</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN CONTENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(f"""
<div class='hero-banner'>
    <div class='hero-title'>🌾 AgroSmart AI</div>
    <div class='hero-subtitle'>
        Intelligent Crop Advisory · Real-time Climate Data · ML-Powered Recommendation
    </div>
    <div style='margin-top:0.8rem; font-size:0.8rem; color:#4b5563;'>
        📍 {st.session_state.city}  ·  👤 {st.session_state.name}
        · {datetime.now().strftime('%d %b %Y, %H:%M')}
    </div>
</div>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD MODEL + FETCH DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.spinner("⚙️  Loading AI model..."):
    predict_proba_fn, classes, model_acc, feat_imp = train_model()
    st.session_state.model_accuracy = model_acc

lat, lon, display_name = geocode(st.session_state.city)

with st.spinner(f"🌐  Fetching live weather for {st.session_state.city}..."):
    weather = fetch_weather(lat, lon)

if weather is None:
    st.error("⚠️ Unable to fetch weather data. Please check your connection.")
    st.stop()

temp     = weather["temp"]
humidity = weather["humidity"]
rain     = weather["rainfall"]

# Soil data
if st.session_state.manual_soil:
    soil = {
        "N":  st.session_state.soil_N,
        "P":  st.session_state.soil_P,
        "K":  st.session_state.soil_K,
        "ph": st.session_state.soil_ph,
    }
    soil_source = "Manual Entry"
else:
    soil = infer_soil_from_region(st.session_state.city, temp, rain, humidity)
    soil_source = "Region-based Estimation"

N, P, K, ph = soil["N"], soil["P"], soil["K"], soil["ph"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEATHER + SOIL DISPLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(f"<div class='section-header'>🌦 Live Weather — {display_name[:50]}</div>", unsafe_allow_html=True)
st.markdown(f"<span class='tag-green'>Source: {weather['source']}</span>", unsafe_allow_html=True)

w1, w2, w3, w4, w5, w6 = st.columns(6)
weather_metrics = [
    (w1, "🌡", f"{temp}°C",           f"Feels {weather['feels_like']}°C", "Temperature"),
    (w2, "💧", f"{humidity}%",         "",                                 "Humidity"),
    (w3, "🌧", f"{rain}mm",            "monthly est.",                     "Rainfall"),
    (w4, "🌬", f"{weather['wind_speed']} km/h", "",                        "Wind"),
    (w5, "📊", f"{weather['pressure']} hPa",    "",                        "Pressure"),
    (w6, "☁",  weather["description"], "",                                 "Condition"),
]

for col, icon, val, sub, label in weather_metrics:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>{icon}</div>
            <div class='metric-value' style='font-size:1.15rem;'>{val}</div>
            <div class='metric-unit'>{sub}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"<div class='section-header'>🌱 Soil Parameters <span style='font-size:0.75rem; color:#4b5563; font-weight:400; margin-left:0.5rem;'>({soil_source})</span></div>",
            unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
soil_metrics = [
    (s1, "🟢", f"{N}", "kg/ha",  "Nitrogen (N)"),
    (s2, "🔵", f"{P}", "kg/ha",  "Phosphorus (P)"),
    (s3, "🟡", f"{K}", "kg/ha",  "Potassium (K)"),
    (s4, "⚗️",  f"{ph}", "0–14",  "Soil pH"),
]
for col, icon, val, unit, label in soil_metrics:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>{icon}</div>
            <div class='metric-value'>{val}</div>
            <div class='metric-unit'>{unit}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ML PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
input_df = pd.DataFrame([[N, P, K, temp, humidity, ph, rain]],
                        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

proba    = predict_proba_fn(input_df)[0]
top_idx  = proba.argsort()[-10:][::-1]
top10    = [(classes[i], round(proba[i] * 100, 1)) for i in top_idx]
best_crop, best_conf = top10[0]

# Save to history
st.session_state.prediction_history.append({
    "city": st.session_state.city,
    "crop": best_crop,
    "conf": best_conf,
    "time": datetime.now().strftime("%H:%M"),
})

crop_info = CROP_DB.get(best_crop, {})
crop_emoji = crop_info.get("emoji", "🌿")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tabs = st.tabs(["🌾 Recommendation", "💰 Economics", "📊 Analytics", "📋 Crop Guide", "🤖 AI Advisor"])
tab_rec, tab_econ, tab_charts, tab_guide, tab_ai = tabs

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — RECOMMENDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_rec:
    col_main, col_side = st.columns([1, 1], gap="large")

    with col_main:
        st.markdown(f"""
        <div class='crop-result-card'>
            <div style='font-size:3.5rem;'>{crop_emoji}</div>
            <div class='crop-name'>{best_crop.upper()}</div>
            <div class='crop-confidence'>🎯 {best_conf}% Confidence</div>
            <div style='margin-top:0.8rem; font-size:0.85rem; color:#6b7280;'>
                Season: {crop_info.get('season','—')} &nbsp;·&nbsp;
                Duration: {crop_info.get('duration_days','—')} days &nbsp;·&nbsp;
                Water: {crop_info.get('water_need','—')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Suitability Analysis
        suit_score, reasons, warnings_list = get_suitability_score(best_crop, temp, rain, ph, humidity)
        st.markdown("<div class='section-header'>🎯 Suitability Analysis</div>", unsafe_allow_html=True)
        st.progress(suit_score / 100)
        st.markdown(f"**Suitability Score: {suit_score}/100**")

        for r in reasons:
            st.markdown(f"<div class='alert-success'>{r}</div>", unsafe_allow_html=True)
        for w in warnings_list:
            st.markdown(f"<div class='alert-warning'>{w}</div>", unsafe_allow_html=True)

    with col_side:
        st.markdown("<div class='section-header'>🌿 Top 10 Recommended Crops</div>", unsafe_allow_html=True)
        for rank, (crop, conf) in enumerate(top10, 1):
            ci = CROP_DB.get(crop, {})
            e = ci.get("emoji", "🌿")
            color = "#4ade80" if rank == 1 else "#22c55e" if rank <= 3 else "#166534"
            st.markdown(f"""
            <div class='rank-badge'>
                <div class='rank-num' style='color:{color};'>#{rank}</div>
                <div style='font-size:1.2rem;'>{e}</div>
                <div class='rank-name'>{crop.title()}</div>
                <div class='rank-bar-bg'>
                    <div class='rank-bar-fill' style='width:{conf}%;'></div>
                </div>
                <div class='rank-pct'>{conf}%</div>
            </div>
            """, unsafe_allow_html=True)

    # Fertilizer recommendation
    if crop_info.get("fertilizer"):
        st.markdown("<div class='section-header'>🧪 Recommended Fertilizer Dosage</div>",
                    unsafe_allow_html=True)
        fert = crop_info["fertilizer"]
        fc1, fc2, fc3 = st.columns(3)
        for col, key, label, color in [
            (fc1, "N", "Nitrogen (N)",    "#4ade80"),
            (fc2, "P", "Phosphorus (P)",  "#60a5fa"),
            (fc3, "K", "Potassium (K)",   "#fbbf24"),
        ]:
            val = fert.get(key, "—")
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-value' style='color:{color};'>{val}</div>
                    <div class='metric-unit'>kg/acre</div>
                    <div class='metric-label'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

    # Pest warnings
    if crop_info.get("pests"):
        st.markdown("<div class='section-header'>⚠️ Common Pests & Diseases</div>", unsafe_allow_html=True)
        pest_html = " ".join([f"<span class='tag-amber'>🪲 {p}</span>" for p in crop_info["pests"]])
        st.markdown(pest_html, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — ECONOMICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_econ:
    econ = compute_economics(best_crop, acres)
    if not econ:
        st.info("Economic data not available for this crop yet.")
    else:
        st.markdown(f"### 💰 Economic Analysis — {best_crop.title()} ({acres} acre{'s' if acres != 1 else ''})")

        e1, e2, e3, e4 = st.columns(4)
        for col, label, val, color, icon in [
            (e1, "Total Revenue",  f"₹{econ['revenue']:,.0f}",     "#4ade80", "💵"),
            (e2, "Total Cost",     f"₹{econ['cost']:,.0f}",         "#f87171", "📉"),
            (e3, "Net Profit",     f"₹{econ['net_profit']:,.0f}",   "#fbbf24", "💰"),
            (e4, "ROI",            f"{econ['roi']}%",                "#a78bfa", "📊"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-icon'>{icon}</div>
                    <div class='metric-value' style='color:{color};'>{val}</div>
                    <div class='metric-label'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("<div class='agro-card'>", unsafe_allow_html=True)
            st.markdown("**📋 Detailed Breakdown**")
            econ_df = pd.DataFrame([
                {"Metric": "Yield",               "Value": f"{econ['yield_q']} quintals"},
                {"Metric": "Market Price",         "Value": f"₹{econ['price_per_q']}/quintal"},
                {"Metric": "Price per kg",         "Value": f"₹{econ['price_per_kg']}/kg"},
                {"Metric": "Break-even Yield",     "Value": f"{econ['breakeven_q']} quintals"},
                {"Metric": "Crop Duration",        "Value": f"{econ['duration']} days"},
                {"Metric": "Daily Profit",         "Value": f"₹{econ['profit_day']}/day"},
            ])
            st.dataframe(econ_df, hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ec2:
            # Revenue vs Cost Donut
            fig_donut = go.Figure(go.Pie(
                labels=["Net Profit", "Cost"],
                values=[max(econ["net_profit"], 0), econ["cost"]],
                hole=0.6,
                marker_colors=["#22c55e", "#374151"],
                textinfo="label+percent",
                textfont=dict(color="white", size=12),
            ))
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Outfit"),
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=False,
                annotations=[dict(text=f"<b>₹{econ['net_profit']:,.0f}</b><br>Profit",
                                  x=0.5, y=0.5, font_size=13, showarrow=False,
                                  font_color="#4ade80")]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Compare top 5 crops economically
        st.markdown("<div class='section-header'>📊 Top Crops — ROI Comparison</div>", unsafe_allow_html=True)
        comp_crops = [c for c, _ in top10[:8] if CROP_DB.get(c)]
        comp_data = []
        for c in comp_crops:
            e = compute_economics(c, acres)
            if e:
                comp_data.append({"Crop": c.title(), "ROI (%)": e["roi"],
                                   "Net Profit (₹)": e["net_profit"],
                                   "Revenue (₹)": e["revenue"]})

        if comp_data:
            cdf = pd.DataFrame(comp_data).sort_values("ROI (%)", ascending=False)
            fig_roi = px.bar(
                cdf, x="Crop", y="ROI (%)",
                color="ROI (%)", color_continuous_scale=["#166534", "#22c55e", "#86efac"],
                text="ROI (%)"
            )
            fig_roi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Outfit"),
                margin=dict(t=20, b=20, l=20, r=20),
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            fig_roi.update_traces(texttemplate="%{text}%", textposition="outside",
                                  marker_line_color="rgba(0,0,0,0)")
            st.plotly_chart(fig_roi, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_charts:
    ac1, ac2 = st.columns(2)

    with ac1:
        # Feature importance
        st.markdown("<div class='section-header'>🔍 Feature Importance</div>", unsafe_allow_html=True)
        fi_df = feat_imp.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fi_df["Feature"] = fi_df["Feature"].replace({
            "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
            "temperature": "Temperature", "humidity": "Humidity",
            "ph": "Soil pH", "rainfall": "Rainfall"
        })
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale=["#166534", "#22c55e", "#4ade80"])
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Outfit"),
            margin=dict(t=10, b=10, l=10, r=10),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with ac2:
        # Radar chart — current conditions vs ideal
        st.markdown("<div class='section-header'>🎯 Current vs Ideal Conditions</div>", unsafe_allow_html=True)
        if crop_info:
            params   = ["Temperature", "Humidity", "Rainfall", "pH", "Nitrogen", "Potassium"]
            t_mid    = sum(crop_info.get("temp_range",   (25, 30))) / 2
            r_mid    = sum(crop_info.get("rain_range",   (80, 120))) / 2
            p_mid    = sum(crop_info.get("ph_range",     (6, 7))) / 2
            fert_mid = crop_info.get("fertilizer", {})

            ideal_raw    = [t_mid,    75,           r_mid,       p_mid,       fert_mid.get("N", 80),  fert_mid.get("K", 50)]
            current_raw  = [temp,     humidity,     rain,        ph,          N,                       K]
            max_vals     = [45,       100,          250,         9,           200,                     700]

            def normalise(vals):
                return [min(v / m * 100, 100) for v, m in zip(vals, max_vals)]

            ideal_n   = normalise(ideal_raw)
            current_n = normalise(current_raw)
            params_c  = params + [params[0]]
            ideal_c   = ideal_n  + [ideal_n[0]]
            current_c = current_n + [current_n[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=ideal_c, theta=params_c, fill="toself", name="Ideal",
                line=dict(color="#22c55e", width=2),
                fillcolor="rgba(34,197,94,0.1)"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=current_c, theta=params_c, fill="toself", name="Current",
                line=dict(color="#f59e0b", width=2, dash="dash"),
                fillcolor="rgba(245,158,11,0.08)"
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1f3320", color="#4b5563"),
                    angularaxis=dict(gridcolor="#1f3320", color="#86efac"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Outfit"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # Crop confidence distribution
    st.markdown("<div class='section-header'>📈 Prediction Confidence Distribution</div>",
                unsafe_allow_html=True)
    conf_df = pd.DataFrame(top10[:10], columns=["Crop", "Confidence"])
    conf_df["Crop"] = conf_df["Crop"].str.title()
    fig_conf = px.bar(conf_df, x="Crop", y="Confidence",
                      color="Confidence",
                      color_continuous_scale=["#0d2010", "#16a34a", "#4ade80"],
                      text="Confidence")
    fig_conf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Outfit"),
        margin=dict(t=20, b=20, l=20, r=20),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Confidence (%)"),
    )
    fig_conf.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig_conf, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — CROP GUIDE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_guide:
    st.markdown("### 📖 Complete Crop Knowledge Base")
    selected_guide = st.selectbox("Select crop to view full guide",
                                  options=list(CROP_DB.keys()),
                                  index=list(CROP_DB.keys()).index(best_crop) if best_crop in CROP_DB else 0,
                                  format_func=lambda c: f"{CROP_DB[c]['emoji']}  {c.title()}")
    g = CROP_DB[selected_guide]
    ge = compute_economics(selected_guide, acres)

    gc1, gc2 = st.columns([1, 1], gap="large")

    with gc1:
        st.markdown("<div class='agro-card'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='font-size:2.5rem; text-align:center;'>{g['emoji']}</div>
        <h2 style='text-align:center; color:#4ade80; margin:0.3rem 0;'>{selected_guide.title()}</h2>
        <div style='text-align:center;'>
            <span class='tag-green'>🌱 {g['season']}</span>
            <span class='tag-green'>💧 {g['water_need']}</span>
            <span class='tag-green'>📅 {g['duration_days']} days</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**🌡 Climate Requirements**")
        st.markdown(f"""
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin:0.5rem 0;'>
            <div class='metric-card' style='padding:0.8rem;'>
                <div class='metric-label'>Temperature</div>
                <div class='metric-value' style='font-size:1rem;'>{g['temp_range'][0]}–{g['temp_range'][1]}°C</div>
            </div>
            <div class='metric-card' style='padding:0.8rem;'>
                <div class='metric-label'>Rainfall</div>
                <div class='metric-value' style='font-size:1rem;'>{g['rain_range'][0]}–{g['rain_range'][1]} mm</div>
            </div>
            <div class='metric-card' style='padding:0.8rem;'>
                <div class='metric-label'>Soil pH</div>
                <div class='metric-value' style='font-size:1rem;'>{g['ph_range'][0]}–{g['ph_range'][1]}</div>
            </div>
            <div class='metric-card' style='padding:0.8rem;'>
                <div class='metric-label'>Best Soil</div>
                <div class='metric-value' style='font-size:0.8rem;'>{g['best_soil']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**🗺 Ideal Growing Regions**")
        region_html = " ".join([f"<span class='tag-green'>📍 {r}</span>" for r in g.get("ideal_regions", [])])
        st.markdown(region_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with gc2:
        # Economics panel
        if ge:
            st.markdown("<div class='agro-card'>", unsafe_allow_html=True)
            st.markdown("**💰 Economics at a Glance**")
            econ_items = [
                ("Input Cost/acre",    f"₹{g['cost_per_acre']:,}"),
                ("Expected Yield",     f"{g['yield_quintals']} quintals"),
                ("Market Price",       f"₹{g['price_per_quintal']}/quintal"),
                ("Net Profit/acre",    f"₹{ge['net_profit']:,.0f}"),
                ("ROI",                f"{ge['roi']}%"),
                ("Break-even Yield",   f"{ge['breakeven_q']} quintals"),
            ]
            for k, v in econ_items:
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; padding:0.4rem 0;
                            border-bottom:1px solid #1f3320; font-size:0.9rem;'>
                    <span style='color:#86efac;'>{k}</span>
                    <span style='font-weight:600; color:#f0fdf4; font-family:"JetBrains Mono",monospace;'>{v}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Agronomic tips
        st.markdown("<div class='agro-card'>", unsafe_allow_html=True)
        st.markdown("**🌿 Expert Agronomic Tips**")
        for tip in g.get("tips", []):
            st.markdown(f"<div class='alert-success'>💡 {tip}</div>", unsafe_allow_html=True)

        st.markdown("**🪲 Watch out for**")
        for pest in g.get("pests", []):
            st.markdown(f"<div class='alert-warning'>⚠ {pest}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — AI ADVISOR (Claude API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ai:
    st.markdown("### 🤖 AI Crop Advisor")
    st.markdown("""
    <div class='alert-success'>
    Ask any farming question. The advisor knows your current location, weather, soil conditions,
    and the ML recommendation — and gives personalised advice.
    </div>
    """, unsafe_allow_html=True)

    anthropic_key = st.text_input(
        "🔑 Enter your Anthropic API key (for AI Advisor)",
        type="password",
        placeholder="sk-ant-...",
        help="Get a key at console.anthropic.com. Your key is not stored."
    )

    context_summary = f"""
Location: {st.session_state.city}
Weather: Temp {temp}°C, Humidity {humidity}%, Rainfall {rain}mm/month, Wind {weather['wind_speed']} km/h
Soil: N={N} kg/ha, P={P} kg/ha, K={K} kg/ha, pH={ph}
ML Recommended Crop: {best_crop.title()} (confidence {best_conf}%)
Top 3 alternatives: {', '.join([c.title() for c, _ in top10[1:4]])}
Farm size: {acres} acres
"""

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display conversation
    for msg in st.session_state.chat_messages:
        role_label = f"👤 {st.session_state.name}" if msg["role"] == "user" else "🤖 AI Advisor"
        color = "#152015" if msg["role"] == "assistant" else "#0d1a0d"
        st.markdown(f"""
        <div style='background:{color}; border:1px solid #1f3320; border-radius:10px;
                    padding:0.8rem 1rem; margin:0.5rem 0;'>
            <div style='font-size:0.75rem; color:#4b5563; margin-bottom:0.3rem;'>{role_label}</div>
            <div style='font-size:0.92rem;'>{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

    user_q = st.text_area("Your question:", placeholder=f"e.g. What are the best practices for growing {best_crop} in {st.session_state.city}?",
                          height=90)

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        ask_btn = st.button("💬 Ask AI", use_container_width=True)
    with col_btn2:
        if st.button("🗑 Clear Chat", use_container_width=False):
            st.session_state.chat_messages = []
            st.rerun()

    if ask_btn and user_q.strip():
        if not anthropic_key.strip():
            st.error("⚠️ Please enter your Anthropic API key to use the AI Advisor.")
        else:
            system_prompt = f"""You are AgroSmart AI Advisor — an expert agricultural consultant for Indian farmers.
You have access to the following real-time farm data:
{context_summary}

Provide concise, actionable advice tailored to the farmer's region and conditions.
Use simple language. Mention specific quantities, timings, and local context where relevant.
If asked about economics, use the crop database values. Be encouraging and practical."""

            messages_payload = [
                {"role": "user", "content": f"Farm context:\n{context_summary}\n\nQuestion: {user_q}"}
                if not st.session_state.chat_messages
                else {"role": "user", "content": user_q}
            ]

            # Include history
            full_msgs = []
            for m in st.session_state.chat_messages:
                full_msgs.append({"role": m["role"], "content": m["content"]})
            full_msgs.append({"role": "user", "content": user_q})

            with st.spinner("🤖 Thinking..."):
                try:
                    resp = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": anthropic_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 800,
                            "system": system_prompt,
                            "messages": full_msgs,
                        },
                        timeout=30
                    )
                    if resp.status_code == 200:
                        answer = resp.json()["content"][0]["text"]
                        st.session_state.chat_messages.append({"role": "user",      "content": user_q})
                        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                        st.rerun()
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text[:200]}")
                except Exception as ex:
                    st.error(f"Request failed: {ex}")

    # Quick question buttons
    st.markdown("<div class='section-header'>⚡ Quick Questions</div>", unsafe_allow_html=True)
    quick_qs = [
        f"What fertilizer schedule should I follow for {best_crop}?",
        f"What are the irrigation requirements for {best_crop} in {st.session_state.city}?",
        f"How can I protect {best_crop} from common diseases?",
        f"When is the best time to sow {best_crop} in my region?",
        "Compare the top 3 recommended crops for profitability",
    ]
    q_cols = st.columns(2)
    for i, q in enumerate(quick_qs):
        with q_cols[i % 2]:
            if st.button(f"❓ {q[:55]}...", key=f"qq_{i}", use_container_width=True):
                if not anthropic_key.strip():
                    st.error("Enter API key first.")
                else:
                    st.session_state.chat_messages.append({"role": "user", "content": q})
                    st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(f"""
<div class='footer-bar'>
    <span>AgroSmart AI</span> · Built with Streamlit, Scikit-learn, Plotly & Anthropic Claude ·
    Real-time weather via AgroMonitoring API ·
    Ensemble ML: RandomForest + GradientBoosting ({model_acc}% accuracy) ·
    19 crops · Region-aware soil estimation
</div>
""", unsafe_allow_html=True)

