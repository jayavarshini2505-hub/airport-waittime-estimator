
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import t, chi2
from datetime import datetime, date, time
from zoneinfo import ZoneInfo


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Airport WaitTime Estimator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    color: #dbeafe;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #64748b;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #0f172a;
}

.info-box {
    padding: 15px;
    border-radius: 12px;
    background-color: #eff6ff;
    border-left: 5px solid #2563eb;
    margin: 15px 0;
    color: black
}

.warning-box {
    padding: 15px;
    border-radius: 12px;
    background-color: #fff7ed;
    border-left: 5px solid #f97316;
    margin: 15px 0;
    color: black;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# AIRPORT DATABASE
# ============================================================

AIRPORTS = {

    "DEL": {
        "airport": "Indira Gandhi International Airport",
        "city": "New Delhi",
        "passengers_million": 79.0,
        "base_wait": 7.0
    },

    "BOM": {
        "airport": "Chhatrapati Shivaji Maharaj International Airport",
        "city": "Mumbai",
        "passengers_million": 55.0,
        "base_wait": 7.0
    },

    "BLR": {
        "airport": "Kempegowda International Airport",
        "city": "Bengaluru",
        "passengers_million": 42.0,
        "base_wait": 7.0
    },

    "HYD": {
        "airport": "Rajiv Gandhi International Airport",
        "city": "Hyderabad",
        "passengers_million": 29.0,
        "base_wait": 7.0
    },

    "MAA": {
        "airport": "Chennai International Airport",
        "city": "Chennai",
        "passengers_million": 22.0,
        "base_wait": 7.0
    },

    "CCU": {
        "airport": "Netaji Subhas Chandra Bose International Airport",
        "city": "Kolkata",
        "passengers_million": 22.0,
        "base_wait": 7.0
    },

    "AMD": {
        "airport": "Sardar Vallabhbhai Patel International Airport",
        "city": "Ahmedabad",
        "passengers_million": 13.0,
        "base_wait": 7.0
    },

    "COK": {
        "airport": "Cochin International Airport",
        "city": "Kochi",
        "passengers_million": 11.0,
        "base_wait": 7.0
    },

    "PNQ": {
        "airport": "Pune Airport",
        "city": "Pune",
        "passengers_million": 10.0,
        "base_wait": 7.0
    }
}


# ============================================================
# AIRPORT DATAFRAME
# ============================================================

airport_df = pd.DataFrame.from_dict(
    AIRPORTS,
    orient="index"
).reset_index()

airport_df.rename(
    columns={"index": "iata"},
    inplace=True
)

airport_df["traffic_rank"] = airport_df[
    "passengers_million"
].rank(pct=True)

airport_df["traffic_factor"] = (
    0.85 +
    0.35 * airport_df["traffic_rank"]
)


# ============================================================
# MODEL FUNCTIONS
# ============================================================

def time_factor(hour):

    if 6 <= hour <= 9:
        return 1.28

    elif 10 <= hour <= 13:
        return 0.98

    elif 14 <= hour <= 17:
        return 0.88

    elif 18 <= hour <= 21:
        return 1.20

    elif 22 <= hour <= 23 or 0 <= hour <= 4:
        return 0.82

    return 1.05


def day_factor(day):

    factors = {
        0: 1.00,
        1: 0.96,
        2: 0.96,
        3: 1.00,
        4: 1.10,
        5: 1.05,
        6: 1.10
    }

    return factors[day]


def month_factor(month):

    if month in [7, 8, 12]:
        return 1.25

    elif month in [4, 10, 11]:
        return 1.10

    return 1.00


def expected_wait(
    airport_code,
    hour,
    selected_date,
    international=False
):

    airport = AIRPORTS[airport_code]

    traffic = airport_df[
        airport_df["iata"] == airport_code
    ]["traffic_factor"].iloc[0]

    wait = (
        airport["base_wait"]
        * traffic
        * time_factor(hour)
        * day_factor(selected_date.weekday())
        * month_factor(selected_date.month)
    )

    if international:
        wait *= 1.12

    return wait


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "airport_waiting_data.csv"
    )


waiting_df = load_data()


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<h1>✈️ Airport WaitTime Estimator</h1>

<p>
Estimate airport security waiting time using
sampling, probability distributions and statistical estimation.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("✈️ Trip Details")

st.sidebar.markdown(
    "Enter your travel details to generate an estimate."
)


# Airport selection

airport_code = st.sidebar.selectbox(

    "Select Airport",

    options=list(AIRPORTS.keys()),

    format_func=lambda x:
        f"{x} — {AIRPORTS[x]['city']}"
)


# Selected airport information

selected_airport = AIRPORTS[airport_code]


# Date selection

selected_date = st.sidebar.date_input(

    "Select Travel Date",

    value=date.today(),

    min_value=date(2026, 1, 1),

    max_value=date(2030, 12, 31)
)


# Time selection

selected_time = st.sidebar.time_input(

    "Select Expected Security Time",

    value=time(8, 0)
)


# International / domestic

international = st.sidebar.radio(

    "Passenger Type",

    ["Domestic", "International"]
)

is_international = (
    international == "International"
)


# Sample size

sample_size = st.sidebar.slider(

    "Statistical Sample Size",

    min_value=30,

    max_value=500,

    value=100,

    step=10
)


# Confidence level

confidence = st.sidebar.selectbox(

    "Confidence Level",

    [0.90, 0.95, 0.99],

    index=1,

    format_func=lambda x:
        f"{int(x * 100)}%"
)


# ============================================================
# MAIN ESTIMATION
# ============================================================

hour = selected_time.hour

model_estimate = expected_wait(

    airport_code,

    hour,

    selected_date,

    is_international
)


# ============================================================
# CREATE RELEVANT STATISTICAL SAMPLE
# ============================================================

airport_population = waiting_df[
    waiting_df["airport_code"] == airport_code
]["waiting_time"]


# Select observations around the requested hour

hour_population = waiting_df[
    (waiting_df["airport_code"] == airport_code)
    &
    (
        abs(
            waiting_df["hour"] - hour
        ) <= 1
    )
]["waiting_time"]


# Fallback if not enough observations

if len(hour_population) < sample_size:

    hour_population = airport_population


sample_n = min(
    sample_size,
    len(hour_population)
)


sample = hour_population.sample(
    n=sample_n,
    random_state=42
)


sample_mean = sample.mean()

sample_std = sample.std(ddof=1)

standard_error = (
    sample_std /
    np.sqrt(sample_n)
)


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

alpha = 1 - confidence

df = sample_n - 1

t_critical = t.ppf(
    1 - alpha / 2,
    df
)

margin_of_error = (
    t_critical *
    standard_error
)

ci_lower = (
    sample_mean -
    margin_of_error
)

ci_upper = (
    sample_mean +
    margin_of_error
)


# ============================================================
# VARIANCE CONFIDENCE INTERVAL
# ============================================================

sample_variance = sample.var(
    ddof=1
)

chi_lower = chi2.ppf(
    alpha / 2,
    df
)

chi_upper = chi2.ppf(
    1 - alpha / 2,
    df
)

variance_lower = (
    df * sample_variance
) / chi_upper

variance_upper = (
    df * sample_variance
) / chi_lower


# ============================================================
# AIRPORT INFORMATION
# ============================================================

st.subheader(
    f"📍 {selected_airport['airport']}"
)

st.write(
    f"**City:** {selected_airport['city']}  |  "
    f"**IATA:** {airport_code}  |  "
    f"**Selected Date:** {selected_date.strftime('%d %B %Y')}  |  "
    f"**Time:** {selected_time.strftime('%I:%M %p')}"
)


# ============================================================
# METRIC CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Estimated Wait",
        f"{model_estimate:.1f} min"
    )


with col2:

    st.metric(
        f"{int(confidence*100)}% Confidence Interval",
        f"{ci_lower:.1f} – {ci_upper:.1f} min"
    )


with col3:

    st.metric(
        "Sample Standard Deviation",
        f"{sample_std:.2f} min"
    )


with col4:

    st.metric(
        "Sample Size",
        f"{sample_n}"
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    f"""
<div class="info-box">

<b>Prediction Summary</b><br><br>

For <b>{selected_airport['city']}</b> on
<b>{selected_date.strftime('%d %B %Y')}</b>
around <b>{selected_time.strftime('%I:%M %p')}</b>,
the statistical model estimates a waiting time of

<b>{model_estimate:.2f} minutes</b>.

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Waiting-Time Distribution",
    "🕐 Hourly Pattern",
    "📐 Sampling & CLT",
    "📋 Statistical Analysis"
])


# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.subheader(
        "Waiting-Time Distribution"
    )

    fig = px.histogram(

        airport_population,

        nbins=35,

        labels={
            "value":
            "Waiting Time (minutes)"
        },

        title=
        f"{selected_airport['city']} Airport Waiting-Time Distribution"
    )


    fig.add_vline(

        x=model_estimate,

        line_dash="dash",

        annotation_text=
        f"Model Estimate: {model_estimate:.2f} min",

        annotation_position="top"
    )


    fig.add_vline(

        x=sample_mean,

        line_dash="dot",

        annotation_text=
        f"Sample Mean: {sample_mean:.2f} min",

        annotation_position="bottom"
    )


    fig.update_layout(
        height=500,
        showlegend=False
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TAB 2
# ============================================================

with tab2:

    st.subheader(
        "Average Waiting Time by Hour"
    )


    hourly_data = (
        waiting_df[
            waiting_df["airport_code"]
            == airport_code
        ]
        .groupby("hour")["waiting_time"]
        .mean()
        .reset_index()
    )


    fig_hour = px.line(

        hourly_data,

        x="hour",

        y="waiting_time",

        markers=True,

        labels={
            "hour": "Hour of Day",
            "waiting_time":
            "Average Waiting Time (minutes)"
        },

        title=
        f"Hourly Waiting-Time Pattern — "
        f"{selected_airport['city']}"
    )


    fig_hour.add_vline(

        x=hour,

        line_dash="dash",

        annotation_text=
        "Selected Time"
    )


    fig_hour.update_layout(
        height=500,
        xaxis=dict(
            dtick=1
        )
    )


    st.plotly_chart(

        fig_hour,

        use_container_width=True
    )


# ============================================================
# TAB 3
# ============================================================

with tab3:

    st.subheader(
        "Central Limit Theorem"
    )


    repetitions = 500

    sample_means = []


    for i in range(repetitions):

        temp_sample = airport_population.sample(

            n=30,

            random_state=i
        )

        sample_means.append(
            temp_sample.mean()
        )


    sample_means = np.array(
        sample_means
    )


    clt_df = pd.DataFrame({
        "Sample Mean":
        sample_means
    })


    fig_clt = px.histogram(

        clt_df,

        x="Sample Mean",

        nbins=30,

        title=
        "Distribution of Repeated Sample Means",

        labels={
            "Sample Mean":
            "Sample Mean Waiting Time (minutes)"
        }
    )


    fig_clt.add_vline(

        x=sample_means.mean(),

        line_dash="dash",

        annotation_text=
        f"Mean = {sample_means.mean():.2f}"
    )


    fig_clt.update_layout(
        height=500
    )


    st.plotly_chart(

        fig_clt,

        use_container_width=True
    )


    st.write(
        f"""
        **500 samples** of size 30 were generated.

        Mean of sample means:
        **{sample_means.mean():.2f} minutes**

        Standard deviation of sample means:
        **{sample_means.std(ddof=1):.3f} minutes**
        """
    )


# ============================================================
# TAB 4
# ============================================================

with tab4:

    st.subheader(
        "Statistical Analysis"
    )


    results_df = pd.DataFrame({

        "Measure": [

            "Airport",

            "Point Estimate",

            "Sample Standard Deviation",

            "Standard Error",

            "Confidence Level",

            "Lower Confidence Limit",

            "Upper Confidence Limit",

            "Sample Variance",

            "Variance Lower Limit",

            "Variance Upper Limit"

        ],

        "Value": [

            f"{airport_code} — "
            f"{selected_airport['city']}",

            f"{sample_mean:.3f} min",

            f"{sample_std:.3f} min",

            f"{standard_error:.3f} min",

            f"{int(confidence*100)}%",

            f"{ci_lower:.3f} min",

            f"{ci_upper:.3f} min",

            f"{sample_variance:.3f} min²",

            f"{variance_lower:.3f} min²",

            f"{variance_upper:.3f} min²"

        ]

    })


    st.dataframe(

        results_df,

        use_container_width=True,

        hide_index=True
    )


# ============================================================
# DATA QUALITY NOTICE
# ============================================================

st.markdown(
    """
<div class="warning-box">

<b>⚠️ Data Quality Notice</b><br><br>

The current waiting-time observations are
model-generated demonstration data based on airport
traffic and time/date factors. They are not live airport
queue measurements.

The application should therefore be interpreted as a
<b>statistical estimation prototype</b>, not a live
operational airport queue system.

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

Airport WaitTime Estimator •
Statistics & Probability Project<br>

Sampling • Point Estimation • Confidence Intervals •
Central Limit Theorem • Variance Estimation

</div>
""",
    unsafe_allow_html=True
)
