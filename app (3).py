import streamlit as st
import pickle

st.set_page_config(page_title="SentiScope", page_icon="🐦", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {background-color: #ffffff;}
.badge-pos {background:#E1F5EE; color:#085041; padding:5px 14px; border-radius:20px; font-weight:600; font-size:15px;}
.badge-neg {background:#FCEBEB; color:#791F1F; padding:5px 14px; border-radius:20px; font-weight:600; font-size:15px;}
.badge-neu {background:#E6F1FB; color:#0C447C; padding:5px 14px; border-radius:20px; font-weight:600; font-size:15px;}
.badge-irr {background:#F1EFE8; color:#444441; padding:5px 14px; border-radius:20px; font-weight:600; font-size:15px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

with st.sidebar:
    st.markdown("## 🐦 SentiScope")
    st.caption("v1.0 · Twitter NLP")
    st.divider()
    page = st.radio("", ["Analyze", "Metrics", "History", "About"], label_visibility="collapsed")

if "history" not in st.session_state:
    st.session_state.history = []

badge_map = {
    "Positive": ("badge-pos", "🟢"),
    "Negative": ("badge-neg", "🔴"),
    "Neutral":  ("badge-neu", "🔵"),
    "Irrelevant": ("badge-irr", "⚪"),
}

if page == "Analyze":
    st.title("Sentiment analyzer")
    st.caption("Paste any tweet to detect Positive · Negative · Neutral · Irrelevant")
    st.divider()
    user_input = st.text_area("TWEET INPUT", placeholder="Type or paste a tweet here...", height=100)
    st.caption(f"{len(user_input)} / 280 characters")
    if st.button("🔍 Analyze", type="primary"):
        if user_input.strip():
            pred = model.predict([user_input])[0]
            proba = model.predict_proba([user_input])[0]
            conf = max(proba) * 100
            cls, icon = badge_map[pred]
            st.markdown("**DETECTED SENTIMENT**")
            st.markdown(f'<span class="{cls}">{icon} {pred}</span>', unsafe_allow_html=True)
            st.write("")
            st.markdown("**CONFIDENCE**")
            st.progress(int(conf))
            st.caption(f"{conf:.1f}% confidence")
            st.session_state.history.insert(0, {
                "tweet": user_input[:60] + "..." if len(user_input) > 60 else user_input,
                "sentiment": pred,
                "confidence": round(conf, 1)
            })
        else:
            st.warning("Pehle kuch text likho!")

elif page == "Metrics":
    st.title("Model performance")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "97.3%")
    col2.metric("F1 Score", "0.97")
    col3.metric("Precision", "0.97")
    col4.metric("Recall", "0.97")
    st.divider()
    import pandas as pd
    st.markdown("**Classification report**")
    st.dataframe(pd.DataFrame({
        "Class": ["Irrelevant","Negative","Neutral","Positive"],
        "Precision": [0.98,0.98,0.98,0.95],
        "Recall": [0.96,0.98,0.97,0.98],
        "F1-Score": [0.97,0.98,0.98,0.96],
        "Support": [172,266,285,277]
    }), use_container_width=True, hide_index=True)

elif page == "History":
    st.title("Recent predictions")
    st.divider()
    if st.session_state.history:
        import pandas as pd
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Abhi koi prediction nahi ki — Analyze tab mein jao!")

elif page == "About":
    st.title("About")
    st.divider()
    st.markdown("""
    **Model**: TF-IDF + Logistic Regression
    **Dataset**: Twitter Entity Sentiment Analysis (Kaggle)
    **Classes**: Positive · Negative · Neutral · Irrelevant
    **Accuracy**: 97.3% on validation set
    **Built with**: Python · Scikit-learn · Streamlit
    """)
