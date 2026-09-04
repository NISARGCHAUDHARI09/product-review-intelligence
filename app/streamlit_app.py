from pathlib import Path
import html
import sys

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "models" / "final_distilbert_model"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Product Review Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HTML RENDERER
# ============================================================

def render_html(content: str):
    """
    Render custom HTML/CSS without displaying HTML source code.
    Streamlit's st.html is preferred.
    """
    if hasattr(st, "html"):
        st.html(content)
    else:
        st.markdown(content, unsafe_allow_html=True)


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>
        .pri-page-title {
            font-size: 2.25rem;
            font-weight: 800;
            color: #f5f7fa;
            margin: 0 0 4px 0;
        }

        .pri-page-subtitle {
            color: #8b949e;
            font-size: 1rem;
            line-height: 1.5;
            margin-bottom: 26px;
        }

        .pri-section-title {
            font-size: 1.25rem;
            font-weight: 750;
            color: #f5f7fa;
            margin: 24px 0 12px 0;
        }

        .pri-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 14px;
            padding: 20px;
            min-height: 142px;
            box-sizing: border-box;
        }

        .pri-card-title {
            color: #8b949e;
            font-size: 0.88rem;
            font-weight: 650;
            margin-bottom: 13px;
        }

        .pri-card-value {
            color: #f5f7fa;
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .pri-card-description {
            color: #8b949e;
            font-size: 0.82rem;
            line-height: 1.4;
            margin-top: 9px;
        }

        .pri-positive {
            color: #3fb950 !important;
        }

        .pri-negative {
            color: #f85149 !important;
        }

        .pri-neutral {
            color: #d29922 !important;
        }

        .pri-review {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 14px;
            padding: 18px 20px;
            color: #c9d1d9;
            line-height: 1.65;
            font-size: 0.95rem;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }

        .pri-aspect {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 14px;
            padding: 18px 20px;
            margin: 12px 0;
        }

        .pri-aspect-name {
            color: #f5f7fa;
            font-size: 1.05rem;
            font-weight: 750;
            margin-bottom: 8px;
        }

        .pri-aspect-sentiment {
            font-size: 0.92rem;
            font-weight: 750;
            margin-bottom: 10px;
        }

        .pri-context {
            color: #9da7b3;
            font-size: 0.9rem;
            line-height: 1.55;
            overflow-wrap: anywhere;
        }

        .pri-context-label {
            color: #c9d1d9;
            font-weight: 700;
        }

        .pri-info {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 15px 18px;
            color: #9da7b3;
            line-height: 1.55;
        }

        .pri-footer {
            text-align: center;
            color: #6e7681;
            font-size: 0.8rem;
            padding: 18px 0 8px 0;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #0b0f14;
        }

        .pri-sidebar-title {
            color: #f5f7fa;
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1.3;
        }

        .pri-sidebar-description {
            color: #8b949e;
            font-size: 0.88rem;
            line-height: 1.5;
            margin-top: 8px;
        }

        /* Text area */
        textarea {
            background: #0d1117 !important;
            color: #f5f7fa !important;
            border: 1px solid #30363d !important;
            border-radius: 10px !important;
        }

        /* Analyze button */
        div.stButton > button {
            width: 100%;
            min-height: 45px;
            border-radius: 10px;
            font-weight: 700;
        }
    </style>
    """
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

try:
    from inference import SentimentPredictor
    from aspect_sentiment import AspectSentimentAnalyzer
except Exception as exc:
    st.error("Could not import the project modules.")
    st.exception(exc)
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading the trained DistilBERT model...")
def load_models():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Model directory not found:\n{MODEL_DIR}"
        )

    predictor = SentimentPredictor(str(MODEL_DIR))
    analyzer = AspectSentimentAnalyzer(predictor)

    return predictor, analyzer


try:
    predictor, absa = load_models()
except Exception as exc:
    st.error("Unable to load the trained DistilBERT model.")
    st.exception(exc)
    st.info(
        "Make sure this folder exists:\n"
        "models/final_distilbert_model/"
    )
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sentiment_class(label):
    value = str(label).upper()

    if value == "POSITIVE":
        return "pri-positive"

    if value == "NEGATIVE":
        return "pri-negative"

    return "pri-neutral"


def sentiment_emoji(label):
    value = str(label).upper()

    if value == "POSITIVE":
        return "😊"

    if value == "NEGATIVE":
        return "😞"

    return "😐"


def percentage(value):
    return f"{float(value) * 100:.2f}%"


def safe_text(value):
    return html.escape(str(value))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html(
        """
        <div class="pri-sidebar-title">
            🎯 Product Review Intelligence
        </div>

        <div class="pri-sidebar-description">
            AI-powered product review analysis using a
            fine-tuned DistilBERT sentiment classifier.
        </div>
        """
    )

    st.divider()

    st.markdown("### Model")
    st.write("**Architecture:** DistilBERT")
    st.write("**Task:** Sentiment Classification")
    st.write("**Classes:** Negative / Neutral / Positive")
    st.write("**Maximum length:** 128 tokens")

    st.divider()

    st.markdown("### Features")
    st.write("✓ Overall sentiment")
    st.write("✓ Confidence score")
    st.write("✓ Sentiment probabilities")
    st.write("✓ Aspect detection")
    st.write("✓ Aspect-level sentiment")
    st.write("✓ Aspect context extraction")


# ============================================================
# HEADER
# ============================================================

render_html(
    """
    <div class="pri-page-title">
        🎯 Product Review Intelligence
    </div>

    <div class="pri-page-subtitle">
        Analyze product reviews using fine-tuned DistilBERT
        and aspect-based sentiment analysis.
    </div>
    """
)


# ============================================================
# INPUT
# ============================================================

render_html(
    '<div class="pri-section-title">📝 Product Review</div>'
)

review = st.text_area(
    "Product Review",
    height=170,
    placeholder=(
        "Example: The camera quality is excellent and the "
        "battery life is disappointing. The screen is beautiful "
        "but the price is too high."
    ),
    label_visibility="collapsed",
)

analyze = st.button(
    "🔍 Analyze Review",
    type="primary",
)


# ============================================================
# EMPTY STATE
# ============================================================

if not review.strip() and not analyze:
    render_html(
        """
        <div class="pri-info">
            <strong>Try an example:</strong><br>
            The camera quality is excellent and the battery life
            is disappointing. The screen is beautiful but the price
            is too high.
        </div>
        """
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if not review.strip():
        st.warning("Please enter a product review.")
        st.stop()

    with st.spinner("Analyzing review..."):
        try:
            prediction = predictor.predict(review)

            label = str(prediction["label"]).upper()
            confidence = float(prediction["confidence"])
            probabilities = prediction["probabilities"]

            result = absa.analyze(review)
            aspects = result.get("aspects", [])

        except Exception as exc:
            st.error("An error occurred while analyzing the review.")
            st.exception(exc)
            st.stop()

    # ========================================================
    # OVERALL SENTIMENT
    # ========================================================

    render_html(
        '<div class="pri-section-title">🎯 Overall Sentiment</div>'
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        css_class = sentiment_class(label)
        emoji = sentiment_emoji(label)

        render_html(
            f"""
            <div class="pri-card">
                <div class="pri-card-title">
                    Predicted Sentiment
                </div>

                <div class="pri-card-value {css_class}">
                    {emoji}&nbsp;&nbsp;{safe_text(label)}
                </div>

                <div class="pri-card-description">
                    Overall review classification
                </div>
            </div>
            """
        )

    with col2:
        render_html(
            f"""
            <div class="pri-card">
                <div class="pri-card-title">
                    Confidence
                </div>

                <div class="pri-card-value">
                    {percentage(confidence)}
                </div>

                <div class="pri-card-description">
                    Confidence of predicted class
                </div>
            </div>
            """
        )

    with col3:
        render_html(
            f"""
            <div class="pri-card">
                <div class="pri-card-title">
                    Detected Aspects
                </div>

                <div class="pri-card-value">
                    {len(aspects)}
                </div>

                <div class="pri-card-description">
                    Product-related aspects detected
                </div>
            </div>
            """
        )

    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    render_html(
        '<div class="pri-section-title">📊 Sentiment Probability</div>'
    )

    probability_rows = []

    for sentiment, probability in probabilities.items():
        probability_rows.append(
            {
                "Sentiment": str(sentiment).upper(),
                "Probability": float(probability),
            }
        )

    probability_df = pd.DataFrame(probability_rows)

    st.bar_chart(
        probability_df.set_index("Sentiment"),
        y="Probability",
    )

    # ========================================================
    # REVIEW
    # ========================================================

    render_html(
        '<div class="pri-section-title">📄 Review</div>'
    )

    render_html(
        f"""
        <div class="pri-review">
            {safe_text(review)}
        </div>
        """
    )

    # ========================================================
    # ASPECT ANALYSIS
    # ========================================================

    render_html(
        '<div class="pri-section-title">🔎 Aspect-Level Sentiment</div>'
    )

    if not aspects:
        st.info(
            "No predefined product aspects were detected."
        )

    else:
        # Aspect table
        aspect_rows = []

        for item in aspects:
            aspect_rows.append(
                {
                    "Aspect": str(item.get("aspect", "")),
                    "Sentiment": str(
                        item.get("sentiment", "")
                    ).upper(),
                    "Confidence": percentage(
                        item.get("confidence", 0)
                    ),
                }
            )

        aspect_df = pd.DataFrame(aspect_rows)

        st.dataframe(
            aspect_df,
            use_container_width=True,
            hide_index=True,
        )

        # Individual aspect cards
        for item in aspects:
            aspect = safe_text(item.get("aspect", ""))
            aspect_label = str(
                item.get("sentiment", "NEUTRAL")
            ).upper()
            aspect_confidence = float(
                item.get("confidence", 0)
            )
            context = safe_text(
                item.get("context", "")
            )

            css_class = sentiment_class(aspect_label)
            emoji = sentiment_emoji(aspect_label)

            render_html(
                f"""
                <div class="pri-aspect">

                    <div class="pri-aspect-name">
                        {aspect}
                    </div>

                    <div class="pri-aspect-sentiment {css_class}">
                        {emoji}&nbsp;&nbsp;{safe_text(aspect_label)}
                        &nbsp;&nbsp;•&nbsp;&nbsp;
                        {percentage(aspect_confidence)}
                    </div>

                    <div class="pri-context">
                        <span class="pri-context-label">
                            Context:
                        </span>
                        {context}
                    </div>

                </div>
                """
            )

    # ========================================================
    # ANALYSIS SUMMARY
    # ========================================================

    render_html(
        '<div class="pri-section-title">📌 Analysis Summary</div>'
    )

    positive_count = sum(
        1
        for item in aspects
        if str(item.get("sentiment", "")).upper() == "POSITIVE"
    )

    neutral_count = sum(
        1
        for item in aspects
        if str(item.get("sentiment", "")).upper() == "NEUTRAL"
    )

    negative_count = sum(
        1
        for item in aspects
        if str(item.get("sentiment", "")).upper() == "NEGATIVE"
    )

    summary1, summary2, summary3 = st.columns(3)

    with summary1:
        render_html(
            f"""
            <div class="pri-card">
                <div class="pri-card-title">
                    Positive Aspects
                </div>

                <div class="pri-card-value pri-positive">
                    {positive_count}
                </div>
            </div>
            """
        )

    with summary2:
        render_html(
            f"""
            <div class="pri-card">
                <div class="pri-card-title">
                    Neutral Aspects
                </div>

                <div class="pri-card-value pri-neutral">
                    {neutral_count}
                </div>
            </div>
            """
        )

    with summary3:
        render_html(
            f"""
            <div class="pri-card">
                <div class="pri-card-title">
                    Negative Aspects
                </div>

                <div class="pri-card-value pri-negative">
                    {negative_count}
                </div>
            </div>
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

render_html(
    """
    <div class="pri-footer">
        Product Review Intelligence
        &nbsp;•&nbsp;
        Fine-tuned DistilBERT
        &nbsp;•&nbsp;
        Aspect-Based Sentiment Analysis
    </div>
    """
)