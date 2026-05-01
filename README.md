# 🌾 AgroSmart AI — Intelligent Crop Advisory System

A production-grade, AI-powered crop recommendation system that combines real-time weather data, region-aware soil estimation, and machine learning to recommend the most profitable crops for any region in India.

---

## ✨ Features

| Feature | Details |
|---|---|
| **ML Model** | Ensemble of RandomForest + GradientBoosting (19 crops, ~95% accuracy) |
| **Real-time Weather** | AgroMonitoring API + OpenWeatherMap fallback + climatological fallback |
| **Smart Soil Engine** | Region-based soil estimation for 40+ Indian cities/districts (N, P, K, pH) |
| **Economic Analysis** | Revenue, cost, ROI, break-even yield, daily profit per acre |
| **AI Advisor** | Claude-powered chat advisor with farm context awareness |
| **Crop Knowledge Base** | 19 crops with fertilizer doses, pest info, expert tips, ideal regions |
| **Analytics** | Feature importance, radar chart, confidence distribution, ROI comparison |

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set API keys as environment variables
export AGRO_API_KEY="your_agromonitoring_key"
export OPENWEATHER_API_KEY="your_openweathermap_key"   # optional fallback

# 4. Run
streamlit run app.py
```

---

## 🧠 ML Architecture

```
Input Features (7)
  ├── N, P, K (soil nutrients, kg/ha)
  ├── Temperature (°C)
  ├── Humidity (%)
  ├── Soil pH
  └── Rainfall (mm/month)

Ensemble Model
  ├── RandomForestClassifier (weight: 55%)
  │     200 trees, max_depth=15
  └── GradientBoostingClassifier (weight: 45%)
        150 estimators, max_depth=5

Output: Crop class + probability distribution (19 crops)
```

### Training Data
- **3,000 synthetic samples** generated from peer-reviewed agronomic literature
- Gaussian distributions centered on each crop's ideal parameters
- Optional: loads `crop_data.csv` if present and merges with synthetic data
- 80/20 train-test split with stratification

---

## 🌍 Crop Database (19 Crops)

Rice · Wheat · Maize · Cotton · Sugarcane · Coffee · Mango · Banana · Tomato · Chickpea · Lentil · Muskmelon · Watermelon · Onion · Groundnut · Soybean · Papaya · Coconut · Turmeric

Each crop includes:
- Temperature, rainfall, pH requirements
- NPK fertilizer doses
- Economic data (cost, yield, market price)
- Expert agronomic tips
- Common pests & diseases
- Ideal growing regions

---

## 📁 Project Structure

```
agrosmart-ai/
├── app.py              ← Main Streamlit application (600+ lines)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── crop_data.csv       ← (Optional) Real dataset to supplement training
```

---

## 🔑 API Keys

| Key | Usage | Free Tier |
|---|---|---|
| `AGRO_API_KEY` | Primary weather data | Yes (limited calls) |
| `OPENWEATHER_API_KEY` | Weather fallback | Yes (1000 calls/day) |
| Anthropic API Key | AI Advisor tab | Pay-per-use |

The app works without API keys using climatological fallback estimates.

---

## 🔭 Future Improvements

- [ ] Satellite NDVI integration for field health monitoring
- [ ] Historical price trends using commodity market APIs
- [ ] Multi-language support (Tamil, Hindi, Telugu)
- [ ] Mobile-responsive PWA version
- [ ] Soil test report PDF upload + parsing
- [ ] Crop calendar with sowing/harvesting reminders
- [ ] Integration with e-NAM (National Agriculture Market)

---

## 👨‍💻 Tech Stack

- **Frontend**: Streamlit + Custom CSS (dark agricultural theme)
- **ML**: Scikit-learn (RandomForest + GradientBoosting ensemble)
- **Visualisation**: Plotly (radar, bar, donut, interactive charts)
- **Weather**: AgroMonitoring REST API
- **Geocoding**: Nominatim (OpenStreetMap)
- **AI Advisor**: Anthropic Claude API

---

*Built as a placement project demonstrating full-stack ML application development.*
