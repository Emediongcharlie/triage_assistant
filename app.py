# app.py

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Load API key from .env or fallback to direct env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Safety check
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Set it in a .env file or system environment.")
    st.stop()

# LangChain setup
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.4,
    api_key=GROQ_API_KEY
)

triage_prompt = PromptTemplate.from_template("""
You are a triage assistant classifying emergency patients into five levels:

1 - Immediate (life-threatening)
2 - Emergent (unstable vitals, severe symptoms)
3 - Urgent (abnormal but stable)
4 - Less Urgent (minor injuries)
5 - Non-Urgent (routine care)

Classify the following:

Symptoms: {symptoms}

Respond only like this:
Triage Level: [1-5] - Detailed Reason
""")

triage_chain = triage_prompt | llm

# Streamlit UI
st.set_page_config(page_title="AI Triage Assistant", page_icon="🩺")
st.title("🩺 AI Triage Assistant")
st.markdown("Enter patient symptoms below to receive a triage level classification.")

with st.form("triage_form"):
    symptoms = st.text_area("📝 Symptoms", height=150, placeholder="E.g. Chest pain and shortness of breath...")
    submitted = st.form_submit_button("Classify")

if submitted:
    if not symptoms.strip():
        st.warning("Please enter some symptoms.")
    else:
        try:
            with st.spinner("Analyzing..."):
                response = triage_chain.invoke({"symptoms": symptoms})
                triage = response.content.strip()

            st.success("✅ Classification Complete:")
            st.markdown(f"**🧠 {triage}**")

            # Save to CSV
            csv_file = "triage_results.csv"
            record = {"Symptoms": symptoms, "Triage Assessment": triage}
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                new_record_df = pd.DataFrame([record])
                df = pd.concat([df, new_record_df], ignore_index=True)
            else:
                df = pd.DataFrame([record])
            df.to_csv(csv_file, index=False)

            st.download_button("⬇ Download Results CSV", data=df.to_csv(index=False),
                               file_name="triage_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f" Error during classification: {e}")
