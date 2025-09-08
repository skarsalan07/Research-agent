import os
import json
import re
import requests
import streamlit as st
from typing import List, Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

------------------------
🔑 API KEYS (Directly Added)
------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyCINDn8dY8Fogn8k1HvAOoWyjCYC5lXNCM"
os.environ["SERPER_API_KEY"] = "d8dcd83951dc429d74b988fe15a4d9dd7bb8bb6a"

------------------------
🤖 LLM: Gemini
------------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

------------------------
🌐 Serper Search Tool
------------------------
def serper_search(query: str) -> str:
url = "https://google.serper.dev/search"
payload = {"q": query}
headers = {
"X-API-KEY": os.environ["SERPER_API_KEY"],
"Content-Type": "application/json"
}
try:
response = requests.post(url, json=payload, headers=headers)
results = response.json()
snippets = []
if "organic" in results:
for item in results["organic"][:5]:
snippets.append(item.get("snippet", ""))
return "\n".join(snippets)
except Exception as e:
return f"⚠️ Serper search failed: {e}"

------------------------
⚙️ Agent State Definition
------------------------
class AgentState(TypedDict):
messages: Annotated[List[Any], add_messages]
company: str
industry: str
key_offerings: List[str]
strategic_focus: List[str]
industry_trends: List[str]
use_cases: List[Dict[str, str]]
datasets: List[Dict[str, str]]
final_report: str

------------------------
🛠 JSON Cleaning Helper
------------------------
def safe_json_parse(response) -> Any:
text = getattr(response, "content", response)
text = re.sub(r"^(json)?|$", "", text.strip(), flags=re.MULTILINE)
try:
return json.loads(text)
except:
return {}

------------------------
🤖 Agents
------------------------
def research_agent(state: AgentState):
company = state["company"]
search_query = f"{company} industry business model products services strategic focus"
search_results = serper_search(search_query)
prompt = f"""
Analyze the following search results about {company} and extract:
1. Industry
2. Key offerings
3. Strategic focus areas

text

Search Results:
{search_results}

Return JSON object with: industry, key_offerings, strategic_focus
"""
response = llm.invoke(prompt)
data = safe_json_parse(response) or {}
return {
    "industry": data.get("industry", "Technology"),
    "key_offerings": data.get("key_offerings", ["Unknown"]),
    "strategic_focus": data.get("strategic_focus", ["Innovation"]),
    "messages": [f"✅ Research completed for {company}"]
}
def trends_agent(state: AgentState):
industry = state["industry"]
search_query = f"{industry} AI ML automation trends 2024 market analysis"
search_results = serper_search(search_query)
prompt = f"""
Based on these results for {industry} AI/ML:
{search_results}
Return JSON list of top 5-7 AI/ML/automation trends.
"""
response = llm.invoke(prompt)
trends = safe_json_parse(response) or [
"AI-powered automation", "Predictive analytics", "Personalization"
]
return {
"industry_trends": trends,
"messages": [f"✅ Trends identified for {industry}"]
}

def use_case_agent(state: AgentState):
company, industry = state["company"], state["industry"]
search_query = f"{company} {industry} AI ML use cases"
search_results = serper_search(search_query)
prompt = f"""
Based on:
- Strategic focus: {state['strategic_focus']}
- Industry trends: {state['industry_trends']}
- Search: {search_results}

text

Generate 3-5 AI/ML use cases in JSON array format with:
title, description, impact, feasibility, required_tech, timeline, roi
"""
response = llm.invoke(prompt)
use_cases = safe_json_parse(response) or [{
    "title": "Predictive Maintenance",
    "description": "Use ML models to predict equipment failures",
    "impact": "Reduce downtime",
    "feasibility": "High",
    "required_tech": ["ML", "IoT", "Analytics"],
    "timeline": "6-12 months",
    "roi": "High"
}]
return {"use_cases": use_cases, "messages": [f"✅ Use cases ready for {company}"]}
def resource_agent(state: AgentState):
industry = state["industry"]
search_query = f"{industry} AI ML datasets Kaggle HuggingFace"
search_results = serper_search(search_query)
prompt = f"""
From these results:
{search_results}
Extract relevant datasets in JSON array: name, url, description, relevance
"""
response = llm.invoke(prompt)
datasets = safe_json_parse(response) or [{
"name": "Example Dataset",
"url": "https://kaggle.com",
"description": "Generic dataset",
"relevance": "Example use"
}]
return {"datasets": datasets, "messages": [f"✅ Datasets collected for {industry}"]}

def evaluator_agent(state: AgentState):
company = state["company"]
prompt = f"""
Prioritize these use cases for {company}:
{state['use_cases']}
Return JSON, add priority_score 1-10 for each.
"""
response = llm.invoke(prompt)
prioritized = safe_json_parse(response) or state["use_cases"]
return {"use_cases": prioritized, "messages": [f"✅ Use cases prioritized for {company}"]}

def final_proposal_agent(state: AgentState):
company, industry = state["company"], state["industry"]
prompt = f"""
Create AI strategy proposal for {company} in {industry}.
Include:
- Executive summary
- Company & industry analysis
- AI/ML trends
- Proposed prioritized use cases
- Resources/datasets
- Recommendations
Format as Markdown.
"""
response = llm.invoke(prompt)
return {"final_report": response.content, "messages": [f"🏆 Report complete for {company}"]}

------------------------
🔗 Workflow
------------------------
def create_workflow():
workflow = StateGraph(AgentState)
workflow.add_node("research", research_agent)
workflow.add_node("trends", trends_agent)
workflow.add_node("usecases", use_case_agent)
workflow.add_node("resources", resource_agent)
workflow.add_node("evaluate", evaluator_agent)
workflow.add_node("final", final_proposal_agent)
workflow.set_entry_point("research")
workflow.add_edge("research", "trends")
workflow.add_edge("trends", "usecases")
workflow.add_edge("usecases", "resources")
workflow.add_edge("resources", "evaluate")
workflow.add_edge("evaluate", "final")
workflow.add_edge("final", END)
return workflow.compile()

workflow = create_workflow()

------------------------
💻 Streamlit UI
------------------------
def generate_use_cases(company_name, progress_area):
state = {
"messages": [], "company": company_name, "industry": "",
"key_offerings": [], "strategic_focus": [],
"industry_trends": [], "use_cases": [],
"datasets": [], "final_report": ""
}

text

# Run entire workflow
result = workflow.invoke(state)

# Display progress manually
with progress_area:
    st.subheader("📌 Research")
    st.write(result["messages"][0])
    st.write("**Industry:**", result["industry"])
    st.write("**Key Offerings:**", result["key_offerings"])
    st.write("**Strategic Focus:**", result["strategic_focus"])

    st.subheader("📌 Trends")
    st.write(result["messages"][1] if len(result["messages"])>1 else "✅ Trends identified")
    st.write("**Industry Trends:**", result["industry_trends"])

    st.subheader("📌 Use Cases")
    st.json(result["use_cases"])

    st.subheader("📌 Datasets")
    st.json(result["datasets"])

    st.subheader("📌 Final Report")
    st.markdown(result["final_report"])

return result
------------------------
🎨 Streamlit App
------------------------
st.set_page_config(page_title="AI Strategy Generator", layout="wide")
st.title("🚀 AI Strategy Report Generator")
company = st.text_input("Enter Company Name:", "Tesla")

if st.button("Generate AI Strategy"):
progress_area = st.container()
result = generate_use_cases(company, progress_area)

text

# Save report and enable download
os.makedirs("reports", exist_ok=True)
filename = f"reports/{company}_ai_strategy_report.md"
with open(filename, "w", encoding="utf-8") as f:
    f.write(result["final_report"])

with open(filename, "r", encoding="utf-8") as f:
    st.download_button("⬇️ Download Report", f, file_name=f"{company}_ai_strategy_report.md")
