import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
OUTPUT_CSV = "triage_results.csv"


def load_llm(api_key: str, model: str):
    return ChatGroq(
        model=model,
        temperature=0.3,
        api_key=api_key
    )


def get_prompt():
    return PromptTemplate.from_template("""
You are a triage assistant classifying emergency patients into five levels:

1 - Immediate (life-threatening)
2 - Emergent (unstable vitals, severe symptoms)
3 - Urgent (abnormal but stable)
4 - Less Urgent (minor injuries)
5 - Non-Urgent (routine care)

Classify the following:

Symptoms: {symptoms}

Respond only like this:
Triage Level: [1-5] - Reason
""")


def classify_symptoms_interactively(llm, prompt):
    chain = prompt | llm
    results = []

    print("🩺 Enter patient symptoms one at a time. Type 'done' to finish.\n")

    while True:
        symptom = input("Enter symptoms: ").strip()
        if symptom.lower() == "done":
            break
        if not symptom:
            continue

        try:
            response = chain.invoke({"symptoms": symptom})
            triage = response.content.strip()
            print(f"{triage}\n")
            results.append({
                "Symptoms": symptom,
                "Triage Assessment": triage
            })
        except Exception as e:
            print(f" Error: {e}")
            results.append({
                "Symptoms": symptom,
                "Triage Assessment": f"Error: {e}"
            })

    return results


def save_results(results, output_path):
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\n Triage results saved to {output_path}")
    else:
        print("\n No symptoms entered.")


def main():
    llm = load_llm(GROQ_API_KEY, MODEL_NAME)
    prompt = get_prompt()
    results = classify_symptoms_interactively(llm, prompt)
    save_results(results, OUTPUT_CSV)


if __name__ == "__main__":
    main()
