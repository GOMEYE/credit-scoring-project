import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO

import sqlite3
from datetime import datetime


st.set_page_config(
    page_title="Credit Risk Scoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def apply_custom_styling():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0E1826;
    }

    h1, h2, h3 {
        font-family: 'Source Serif 4', serif !important;
        color: #E7ECF3 !important;
        letter-spacing: -0.01em;
    }

    /* Eyebrow label style, used via markdown */
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #C9A227;
        margin-bottom: 0.25rem;
    }

    /* Hairline divider */
    hr {
        border-color: #2C3E5C !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #C9A227;
        color: #0E1826;
        border: none;
        border-radius: 4px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.5rem 1.5rem;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #E0B838;
        color: #0E1826;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid #2C3E5C;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        color: #93A2BC;
        text-transform: uppercase;
    }
    .stTabs [aria-selected="true"] {
        color: #C9A227 !important;
        border-bottom: 2px solid #C9A227 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #E7ECF3 !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        color: #93A2BC !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em;
    }

    /* Progress bar (confidence) */
    .stProgress > div > div > div {
        background-color: #C9A227 !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        font-family: 'IBM Plex Mono', monospace;
        border: 1px solid #2C3E5C;
        border-radius: 4px;
    }

    /* Input widgets */
    .stSelectbox > div > div, .stNumberInput > div > div {
        background-color: #16233A;
        border: 1px solid #2C3E5C;
        border-radius: 4px;
        color: #E7ECF3;
    }
                
        div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #2C3E5C !important;
        background-color: #16233A !important;
        border-radius: 6px !important;
    }
                
    /* Multiselect tags */
    span[data-baseweb="tag"] {
        background-color: #C9A227 !important;
        color: #0E1826 !important;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_styling()

DB_PATH = "../data/predictions_log.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_used TEXT,
            age INTEGER,
            sex TEXT,
            job INTEGER,
            housing TEXT,
            saving_accounts TEXT,
            checking_accounts TEXT,
            credit_amount INTEGER,
            duration INTEGER,
            prediction TEXT,
            probability_good REAL,
            credit_score INTEGER,
            decision TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(model_name, age, sex, job, housing, saving_accounts, checking_accounts,
                    credit_amount, duration, prediction, prob_good, credit_score, decision):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, model_used, age, sex, job, housing, saving_accounts,
            checking_accounts, credit_amount, duration, prediction,
            probability_good, credit_score, decision
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(), model_name, age, sex, job, housing,
        saving_accounts, checking_accounts, credit_amount, duration,
        prediction, float(prob_good), credit_score, decision
    ))
    conn.commit()
    conn.close()

init_db()

def generate_pdf_report(input_df, pred, prob_good, credit_score, importance_df, fig, applicant_inputs):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=20, spaceAfter=6)
    heading_style = styles['Heading2']
    normal_style = styles['Normal']

    # Title
    story.append(Paragraph("Credit Risk Assessment Report", title_style))
    story.append(Paragraph("Generated by Credit Risk Scoring System", normal_style))
    story.append(Spacer(1, 20))

    # Applicant details table
    story.append(Paragraph("Applicant Information", heading_style))
    applicant_data = [["Field", "Value"]] + [[k, str(v)] for k, v in applicant_inputs.items()]
    applicant_table = Table(applicant_data, colWidths=[200, 250])
    applicant_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(applicant_table)
    story.append(Spacer(1, 20))

    # Prediction result
    story.append(Paragraph("Prediction Result", heading_style))
    result_text = "Good" if pred == 1 else "Bad"
    decision_text = "APPROVED" if credit_score >= 700 else ("CONDITIONAL APPROVAL" if credit_score >= 550 else "DENIED")

    result_style = ParagraphStyle('Result', parent=normal_style, fontSize=13, spaceAfter=4)
    story.append(Paragraph(f"<b>Predicted Credit Risk:</b> {result_text}", result_style))
    story.append(Paragraph(f"<b>Credit Score:</b> {credit_score} ({decision_text})", result_style))
    story.append(Paragraph(f"<b>Probability of being a reliable borrower:</b> {prob_good * 100:.1f}%", result_style))
    story.append(Spacer(1, 20))

    # Feature importance table
    story.append(Paragraph("Model Decision Factors (Global Importance)", heading_style))
    imp_data = [["Feature", "Importance"]] + [
        [row['Feature'], f"{row['Importance']:.4f}"] for _, row in importance_df.iterrows()
    ]
    imp_table = Table(imp_data, colWidths=[250, 150])
    imp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(imp_table)
    story.append(Spacer(1, 20))

   # SHAP waterfall chart
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)

    story.append(KeepTogether([
        Paragraph("Local Explainability: Why This Decision Was Made", heading_style),
        Spacer(1, 8),
        Image(img_buffer, width=6.5*inch, height=4*inch)
    ]))

    doc.build(story)
    buffer.seek(0)
    return buffer

AVAILABLE_MODELS = {
    "XGBoost (Recommended)": "../models/xgboost_balanced_credit_model.pkl",
    "Random Forest": "../models/random_forest_balanced.pkl",
    "Extra Trees": "../models/extra_trees_balanced.pkl",
    "Logistic Regression": "../models/logistic_regression_balanced.pkl",
    "SVM": "../models/svm_balanced.pkl",
}

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# tabs layout at the top of the app
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Credit Risk Predictor", "📊 Historical Data Insights", "📁 Batch Prediction", "🗂️ Prediction Log"])

encoders = {col: joblib.load(f"../models/{col}_encoder.pkl") for col in ('Sex', 'Housing', 'Saving accounts', 'Checking account')}
background_data = joblib.load('../models/shap_background.pkl')

with tab1:
    st.markdown(
        """
        <div class="eyebrow">Credit Risk Scoring System</div>
        <h1 style="margin-top:0.25rem; margin-bottom:0.25rem;">Assess an Application</h1>
        <p style="color:#93A2BC; font-size:1rem; margin-bottom:1.5rem;">
            Enter an applicant's details to generate a risk assessment, credit score, and explanation.
        </p>
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Model</div>', unsafe_allow_html=True)
        selected_model_name = st.selectbox(
            "Choose which trained model to use for prediction:",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
            label_visibility="collapsed",
            help="XGBoost is recommended based on evaluation results (best F1-macro and recall on high-risk applicants)."
        )
        st.caption(f"Currently using **{selected_model_name}**")

    model = load_model(AVAILABLE_MODELS[selected_model_name])

    if selected_model_name in ["XGBoost (Recommended)", "Random Forest", "Extra Trees"]:
        explainer = shap.TreeExplainer(model, model_output="probability", data=background_data)
        shap_available = True
    else:
        shap_available = False

    st.markdown("---")

    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
    housing = st.selectbox("Housing", ['own', 'rent', 'free'])
    saving_accounts = st.selectbox("Saving Accounts", ['little', 'moderate', 'rich', 'quite rich'], index=1)
    checking_accounts = st.selectbox("Checking Accounts", ['little', 'moderate', 'rich', 'quite rich'], index=1)
    credit_amount = st.number_input("Credit Amount", min_value=0, value=2500)
    duration = st.number_input("Duration (months)", min_value=1, value=18)

    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [encoders["Sex"].transform([sex])[0]],
        "Job": [job],
        "Housing": [encoders["Housing"].transform([housing])[0]],
        "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
        "Checking account": [encoders["Checking account"].transform([checking_accounts])[0]],
        "Credit amount": [credit_amount],
        "Duration": [duration]
    })

    if st.button("Predict Credit Risk"):
        # 1. Get the raw prediction (0 or 1)
        pred = model.predict(input_df)[0]

        # 2. Get the probability scores for the 3-digit credit score
        # predict_proba returns [prob_of_bad, prob_of_good]
        probabilities = model.predict_proba(input_df)[0]
        prob_good = probabilities[1]  # Chance that they are a good borrower

        # Map probability to a standard credit score scale (300 to 850)
        credit_score = int(300 + (prob_good * 550))

        decision_str = "Approved" if credit_score >= 700 else ("Conditional" if credit_score >= 550 else "Denied")
        result_str = "Good" if pred == 1 else "Bad"
        status_colors = {"Approved": "#3FA796", "Conditional": "#D9A441", "Denied": "#C1523D"}
        status_color = status_colors[decision_str]
        score_label = {"Approved": "Excellent", "Conditional": "Fair", "Denied": "Poor Risk"}[decision_str]

        st.markdown(
            f"""
            <div style="background-color:#16233A; border:1px solid {status_color}; border-radius:8px;
                        padding:1.5rem 2rem; margin-top:1rem; display:flex; align-items:center;
                        justify-content:space-between; flex-wrap:wrap; gap:1.5rem;">
                <div>
                    <div class="eyebrow" style="color:{status_color};">Risk Assessment</div>
                    <div style="font-family:'Source Serif 4', serif; font-size:1.8rem; color:#E7ECF3; margin-top:0.25rem;">
                        {result_str} Credit Risk
                    </div>
                    <div style="font-family:'IBM Plex Mono', monospace; color:#93A2BC; font-size:0.9rem; margin-top:0.5rem;">
                        Probability of reliable repayment: <span style="color:#E7ECF3;">{prob_good*100:.1f}%</span>
                    </div>
                </div>
                <div style="text-align:center;">
                    <div style="font-family:'IBM Plex Mono', monospace; font-size:2.75rem; font-weight:600; color:{status_color}; line-height:1;">
                        {credit_score}
                    </div>
                    <div class="eyebrow" style="margin-top:0.25rem;">Credit Score · {score_label}</div>
                </div>
                <div style="border:2px solid {status_color}; color:{status_color}; padding:0.6rem 1.2rem;
                            border-radius:4px; font-family:'IBM Plex Mono', monospace; font-weight:600;
                            letter-spacing:0.1em; text-transform:uppercase; transform:rotate(-3deg);">
                    {decision_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------------------------------------
        # CONFIDENCE / UNCERTAINTY DISPLAY
        # -------------------------------------------------------------
        st.markdown("---")

        confidence_pct = float(abs(prob_good - 0.5) * 2 * 100)

        if confidence_pct >= 70:
            confidence_label = "High Confidence"
            tier_color = "#3FA796"
        elif confidence_pct >= 35:
            confidence_label = "Moderate Confidence"
            tier_color = "#D9A441"
        else:
            confidence_label = "Low Confidence — borderline case"
            tier_color = "#C1523D"

        st.markdown(
            f"""
            <div class="eyebrow">Prediction Confidence</div>
            <p style="color:{tier_color}; font-family:'IBM Plex Mono', monospace; font-size:0.95rem;
                       font-weight:600; margin-top:0.25rem; margin-bottom:0.5rem;">
                {confidence_label} — {confidence_pct:.1f}%
            </p>
            <div style="background-color:#16233A; border:1px solid #2C3E5C; border-radius:4px;
                        height:10px; width:100%; overflow:hidden;">
                <div style="background-color:{tier_color}; height:100%; width:{confidence_pct}%;
                            border-radius:4px;"></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if confidence_pct < 35:
            st.markdown(
                f"""
                <div style="background-color:#16233A; border-left:3px solid {tier_color}; border-radius:4px;
                            padding:0.75rem 1rem; margin-top:1rem; color:#E7ECF3; font-size:0.9rem;">
                    ⚠️ This prediction is close to the decision boundary. The model is not strongly confident
                    in either direction — consider requesting additional applicant information or manual review.
                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------------------
        # FEATURE IMPORTANCE SECTION (CS Project Requirement)
        # -------------------------------------------------------------
        if shap_available:
         st.markdown("---")
        st.markdown(
            """
            <div class="eyebrow">Model Decision Factors</div>
            <p style="color:#93A2BC; font-size:0.9rem; margin-top:0.25rem;">
                Which factors this model relies on most, across all applicants — not just this one.
            </p>
            """,
            unsafe_allow_html=True
        )

        importance_df = pd.DataFrame({
            'Feature': input_df.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.markdown('<div style="background-color:#16233A; border:1px solid #2C3E5C; border-radius:6px; padding:1rem;">', unsafe_allow_html=True)
        st.bar_chart(data=importance_df, x='Feature', y='Importance', color="#C9A227", height=800)
        st.markdown('</div>', unsafe_allow_html=True)



        # ------------------------------------------------------------------
        # SHAP LOCAL EXPLAINABILITY VISUALIZATION (Waterfall Chart)
        # ------------------------------------------------------------------
        if shap_available:
            st.markdown("---")
            st.subheader("🔍 Local Explainability: Why this decision was made")

            # 1. Compute local SHAP values for the specific applicant row
            shap_values = explainer(input_df)

            # 2. Create a Matplotlib figure environment (Slightly narrower for side-by-side view)
            fig, ax = plt.subplots(figsize=(8, 5))

            if len(shap_values.shape) == 3:  # If multi-class outputs exist
                shap.plots.waterfall(shap_values[0, :, 1], show=False)
            else:
                shap.plots.waterfall(shap_values[0], show=False)

            plt.tight_layout()

            # 3. Create two side-by-side layout columns (1 part text, 1.5 parts chart)
            col_text, col_chart = st.columns([1, 1.5])

            # Left Column: The Technical Explanation
            with col_text:
                    st.markdown(
                        """
                        <div style="background-color: #16233A; border: 1px solid #2C3E5C; border-radius: 6px; padding: 1.25rem 1.5rem;">
                            <div class="eyebrow">How to read this</div>
                            <p style="color: #E7ECF3; font-size: 0.95rem; line-height: 1.5; margin-top: 0.5rem;">
                                Every applicant starts at the average approval chance across all past applicants — about
                                <strong>50%</strong>. From there, each factor listed on the left either pushes the applicant's
                                score <strong>up</strong> (more likely approved) or <strong>down</strong> (more likely denied),
                                based on this specific application.
                            </p>
                            <p style="color: #E7ECF3; font-size: 0.95rem; line-height: 1.5;">
                                <span style="color:#C1523D; font-weight:600;">🔴 Red bars</span> pushed the score <strong>up</strong>
                                (toward approval).<br>
                                <span style="color:#3FA796; font-weight:600;">🔵 Blue bars</span> pushed the score <strong>down</strong>
                                (toward denial).
                            </p>
                            <p style="color: #93A2BC; font-size: 0.85rem; font-style: italic; margin-bottom: 0;">
                                The factors are ordered by how much impact they had on this specific applicant — not
                                how important they are in general.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Right Column: The Visual Chart
            with col_chart:
                st.pyplot(fig)

            # ------------------------------------------------------------------
            # PDF REPORT EXPORT
            # ------------------------------------------------------------------
            st.markdown("---")
            st.subheader("📄 Download Report")

            applicant_inputs = {
                "Age": age, "Sex": sex, "Job": job, "Housing": housing,
                "Saving Accounts": saving_accounts, "Checking Account": checking_accounts,
                "Credit Amount": credit_amount, "Duration (months)": duration
            }

            pdf_buffer = generate_pdf_report(
                input_df, pred, prob_good, credit_score, importance_df, fig, applicant_inputs
            )

            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_buffer,
                file_name=f"credit_risk_report_{age}yo_{sex}.pdf",
                mime="application/pdf"
            )
        else:
            st.info(f"ℹ️ SHAP local explainability and PDF export are currently only available for tree-based models (XGBoost, Random Forest, Extra Trees). {selected_model_name} doesn't support the same explainer type.")


with tab2:
    st.markdown(
        """
        <div class="eyebrow">Credit Risk Scoring System</div>
        <h1 style="margin-top:0.25rem; margin-bottom:0.25rem;">Historical Data</h1>
        <p style="color:#93A2BC; font-size:1rem; margin-bottom:1.5rem;">
            Explore trends and distributions from the training dataset.
        </p>
        """,
        unsafe_allow_html=True
    )

    # 1. Load the raw dataset for exploration
    @st.cache_data
    def load_eda_data():
        df_raw = pd.read_csv("../data/german_credit_data.csv")
        if 'Unnamed: 0' in df_raw.columns:
            df_raw.drop(columns=['Unnamed: 0'], inplace=True)
        return df_raw

    try:
        df_eda = load_eda_data()

        # 2. Key Performance Metrics Row
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Historical Records", f"{len(df_eda)}")
            with col2:
                st.metric("Average Loan Duration", f"{df_eda['Duration'].mean():.1f} Months")
            with col3:
                st.metric("Average Credit Amount", f"${df_eda['Credit amount'].mean():.2f}")

        st.markdown("---")

        # 3. Interactive Filtering Sidebar/Controls
        st.markdown('<div class="eyebrow">Interactive Dataset Filtering</div>', unsafe_allow_html=True)

        # Create filtering inputs side-by-side
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            st.markdown('<div class="eyebrow">Filter by Housing Type</div>', unsafe_allow_html=True)
            selected_housing = st.multiselect(
                "Filter by Housing Type:",
                options=df_eda['Housing'].dropna().unique(),
                default=df_eda['Housing'].dropna().unique(),
                label_visibility="collapsed"
            )
        with filter_col2:
            st.markdown('<div class="eyebrow">Filter by Age Range</div>', unsafe_allow_html=True)
            age_range = st.slider(
                "Filter by Age Range:",
                int(df_eda['Age'].min()),
                int(df_eda['Age'].max()),
                (int(df_eda['Age'].min()), int(df_eda['Age'].max())),
                label_visibility="collapsed"
            )

        # Apply the filters to the dataframe
        filtered_df = df_eda[
            (df_eda['Housing'].isin(selected_housing)) &
            (df_eda['Age'] >= age_range[0]) &
            (df_eda['Age'] <= age_range[1])
        ]

        # 4. Interactive Visualizations
        st.markdown('<div class="eyebrow" style="margin-top:1.5rem;">Filtered Distribution Insights</div>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown('<div class="eyebrow">Credit Amount Distribution</div>', unsafe_allow_html=True)
            st.bar_chart(filtered_df['Credit amount'].value_counts().sort_index(), color="#C9A227")

        with chart_col2:
            st.markdown('<div class="eyebrow">Purpose of Loans</div>', unsafe_allow_html=True)
            purpose_counts = filtered_df['Purpose'].value_counts()
            st.bar_chart(purpose_counts, color="#3FA796")

        # 5. Raw Data Preview
        with st.expander("📋 View Filtered Raw Data Sub-table"):
            st.dataframe(filtered_df, use_container_width=True)

    except FileNotFoundError:
        st.error("⚠️ 'german_credit_data.csv' not found. Please ensure the dataset file is in your project folder to display dashboard insights.")

with tab3:
    st.markdown(
        """
        <div class="eyebrow">Credit Risk Scoring System</div>
        <h1 style="margin-top:0.25rem; margin-bottom:0.25rem;">Batch Prediction</h1>
        <p style="color:#93A2BC; font-size:1rem; margin-bottom:1.5rem;">
            Upload a CSV of multiple applicants to score them all at once.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    **Required columns:** `Age`, `Sex`, `Job`, `Housing`, `Saving accounts`, `Checking account`, `Credit amount`, `Duration`

    - `Sex`: male / female
    - `Housing`: own / rent / free
    - `Saving accounts` / `Checking account`: little / moderate / rich / quite rich
    - `Job`: integer 0-3
    """)

    uploaded_file = st.file_uploader("Upload applicant data (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            required_cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
                              'Checking account', 'Credit amount', 'Duration']
            missing_cols = [col for col in required_cols if col not in batch_df.columns]

            if missing_cols:
                st.error(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success(f"✅ Loaded {len(batch_df)} applicant records")

                with st.expander("📋 Preview uploaded data"):
                    st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("Run Batch Predictions"):
                    # Encode categorical columns using the same encoders as the single-prediction tab
                    encoded_df = batch_df.copy()
                    for col in ('Sex', 'Housing', 'Saving accounts', 'Checking account'):
                        encoded_df[col] = encoders[col].transform(batch_df[col])

                    # Keep only the model's expected feature columns, in the correct order
                    model_input = encoded_df[required_cols]

                    # Predict
                    predictions = model.predict(model_input)
                    probabilities = model.predict_proba(model_input)[:, 1]
                    credit_scores = (300 + probabilities * 550).astype(int)

                    # Build results
                    results_df = batch_df.copy()
                    results_df['Predicted Risk'] = ['Good' if p == 1 else 'Bad' for p in predictions]
                    results_df['Probability (Good)'] = probabilities.round(3)
                    results_df['Credit Score'] = credit_scores
                    results_df['Decision'] = results_df['Credit Score'].apply(
                        lambda s: 'Approved' if s >= 700 else ('Conditional' if s >= 550 else 'Denied')
                    )

                    st.markdown("---")
                    st.subheader("📊 Batch Results Summary")

                    with st.container(border=True):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Applicants", len(results_df))
                        with col2:
                            st.metric("Approved", (results_df['Decision'] == 'Approved').sum())
                        with col3:
                            st.metric("Conditional", (results_df['Decision'] == 'Conditional').sum())
                        with col4:
                            st.metric("Denied", (results_df['Decision'] == 'Denied').sum())

                    st.markdown("---")
                    st.subheader("📋 Full Results")
                    st.dataframe(results_df, use_container_width=True)

                    # Download button
                    csv_output = results_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv_output,
                        file_name="credit_risk_batch_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

with tab4:
    st.markdown(
        """
        <div class="eyebrow">Credit Risk Scoring System</div>
        <h1 style="margin-top:0.25rem; margin-bottom:0.25rem;">Prediction Log</h1>
        <p style="color:#93A2BC; font-size:1rem; margin-bottom:1.5rem;">
            A complete audit trail of every prediction made through this application.
        </p>
        """,
        unsafe_allow_html=True
    )

    conn = sqlite3.connect(DB_PATH)
    log_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()

    if len(log_df) == 0:
        st.info("No predictions logged yet. Make a prediction in the 'Credit Risk Predictor' tab to see it appear here.")
    else:
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions Logged", len(log_df))
            with col2:
                st.metric("Approved", (log_df['decision'] == 'Approved').sum())
            with col3:
                st.metric("Denied", (log_df['decision'] == 'Denied').sum())

        st.markdown("---")
        st.dataframe(log_df, use_container_width=True)

        csv_output = log_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Full Log as CSV",
            data=csv_output,
            file_name="prediction_log.csv",
            mime="text/csv"
        )