import os
import json
import re
import requests
import streamlit as st
from typing import List, Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------------
# 🔑 API KEYS (Directly Added)
# ------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyCINDn8dY8Fogn8k1HvAOoWyjCYC5lXNCM"
os.environ["SERPER_API_KEY"] = "d8dcd83951dc429d74b988fe15a4d9dd7bb8bb6a"


# ------------------------
# 🤖 LLM: Gemini
# ------------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# ------------------------
# 🌐 Serper Search Tool
# ------------------------
def serper_search(query: str, num_results=5):
    url = "https://google.serper.dev/search"
    payload = {"q": query}
    headers = {"X-API-KEY": os.environ["SERPER_API_KEY"], "Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers).json()
        results = []
        if "organic" in response:
            for item in response["organic"][:num_results]:
                results.append({
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "link": item.get("link")
                })
        return results
    except Exception as e:
        return []

# ------------------------
# ⚙️ Agent State Definition
# ------------------------
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    company: str
    industry: str
    key_offerings: List[str]
    strategic_focus: List[str]
    industry_trends: List[str]
    use_cases: List[Dict[str, str]]
    datasets: List[Dict[str, str]]
    competitors: List[Dict[str, str]]
    final_report: str
    citations: List[str]

# ------------------------
# 🛠 JSON Cleaning Helper
# ------------------------
def safe_json_parse(response) -> Any:
    text = getattr(response, "content", response)
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except:
        return {}

# ------------------------
# 🤖 Agents
# ------------------------
def research_agent(state: AgentState):
    company = state["company"]
    search_results = serper_search(f"{company} industry business model products services strategic focus")
    prompt = f"""
    Analyze search results about {company}. Extract:
    1. Industry
    2. Key offerings
    3. Strategic focus areas
    Include structured JSON with citations from URLs.
    Search Results: {search_results}
    """
    response = llm.invoke(prompt)
    data = safe_json_parse(response) or {}
    citations = [r["link"] for r in search_results]
    return {
        "industry": data.get("industry", "Technology"),
        "key_offerings": data.get("key_offerings", ["Unknown"]),
        "strategic_focus": data.get("strategic_focus", ["Innovation"]),
        "messages": [f"✅ Research completed for {company}"],
        "citations": citations
    }

def competitor_agent(state: AgentState):
    company = state["company"]
    search_results = serper_search(f"{company} competitors annual reports")
    competitors = [{"name": r["title"], "report": r["link"]} for r in search_results]
    return {"competitors": competitors, "messages": [f"✅ Competitors identified for {company}"]}

def trends_agent(state: AgentState):
    industry = state["industry"]
    search_results = serper_search(f"{industry} AI ML automation trends 2024 market analysis")
    prompt = f"""
    Based on results for {industry} AI/ML, return top 5-7 trends in structured JSON with citations.
    Search Results: {search_results}
    """
    response = llm.invoke(prompt)
    trends = safe_json_parse(response) or ["AI automation", "Predictive analytics", "Personalization"]
    citations = [r["link"] for r in search_results]
    return {"industry_trends": trends, "messages": [f"✅ Trends identified"], "citations": citations}

def use_case_agent(state: AgentState):
    company, industry = state["company"], state["industry"]
    search_results = serper_search(f"{company} {industry} AI ML use cases")
    prompt = f"""
    Generate 3-5 AI/ML use cases in JSON array:
    title, description, impact, feasibility, required_tech, timeline, roi
    Include citations from URLs
    """
    response = llm.invoke(prompt)
    use_cases = safe_json_parse(response) or []
    citations = [r["link"] for r in search_results]
    return {"use_cases": use_cases, "messages": [f"✅ Use cases ready"], "citations": citations}

def resource_agent(state: AgentState):
    industry = state["industry"]
    search_results = serper_search(f"{industry} AI ML datasets site:kaggle.com OR site:huggingface.co")
    datasets = [{"name": r["title"], "url": r["link"], "license": "Unknown", "relevance": "High"} for r in search_results]
    # Save resources.md
    os.makedirs("reports", exist_ok=True)
    with open("reports/resources.md", "w", encoding="utf-8") as f:
        for ds in datasets:
            f.write(f"- [{ds['name']}]({ds['url']}) | {ds['license']} | {ds['relevance']}\n")
    citations = [r["link"] for r in search_results]
    return {"datasets": datasets, "messages": [f"✅ Resources collected"], "citations": citations}

def evaluator_agent(state: AgentState):
    for uc in state["use_cases"]:
        uc["priority_score"] = min(10, len(uc.get("required_tech", []))*2 + len(uc.get("impact", ""))//5)
    return {"use_cases": state["use_cases"], "messages": [f"✅ Use cases prioritized"]}

def final_proposal_agent(state: AgentState):
    flowchart_code = """
    graph TD
        A[Research] --> B[Competitors]
        B --> C[Trends]
        C --> D[Use Cases]
        D --> E[Resources]
        E --> F[Evaluator]
        F --> G[Final Report]
    """
    prompt = f"""
    Generate full AI strategy proposal for {state['company']} in {state['industry']}:
    Include references, citations, competitors, datasets (clickable), and embed flowchart.
    """
    response = llm.invoke(prompt)
    return {"final_report": response.content + f"\n\n```mermaid\n{flowchart_code}\n```", "messages": [f"🏆 Final report complete"]}

# ------------------------
# 🔗 Workflow
# ------------------------
def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("research", research_agent)
    workflow.add_node("competitors", competitor_agent)
    workflow.add_node("trends", trends_agent)
    workflow.add_node("usecases", use_case_agent)
    workflow.add_node("resources", resource_agent)
    workflow.add_node("evaluate", evaluator_agent)
    workflow.add_node("final", final_proposal_agent)
    workflow.set_entry_point("research")
    workflow.add_edge("research", "competitors")
    workflow.add_edge("competitors", "trends")
    workflow.add_edge("trends", "usecases")
    workflow.add_edge("usecases", "resources")
    workflow.add_edge("resources", "evaluate")
    workflow.add_edge("evaluate", "final")
    workflow.add_edge("final", END)
    return workflow.compile()

workflow = create_workflow()

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="AI Strategy Generator", layout="wide")
st.title("🚀 AI Strategy Report Generator")
company = st.text_input("Enter Company Name:", "Tesla")

def generate_use_cases(company_name, progress_area):
    state = {
        "messages": [], "company": company_name, "industry": "",
        "key_offerings": [], "strategic_focus": [],
        "industry_trends": [], "use_cases": [],
        "datasets": [], "competitors": [], "final_report": "", "citations": []
    }
    for step in ["research", "competitors", "trends", "usecases", "resources", "evaluate", "final"]:
        state.update(workflow.nodes[step](state))
        with progress_area:
            st.subheader(f"📌 Step: {step.capitalize()}")
            st.write(state["messages"][-1])
            if step == "research":
                st.write("**Industry:**", state["industry"])
                st.write("**Key Offerings:**", state["key_offerings"])
                st.write("**Strategic Focus:**", state["strategic_focus"])
                for c in state["citations"]:
                    st.markdown(f"- [{c}]({c})")
            elif step == "competitors":
                for c in state["competitors"]:
                    st.markdown(f"- {c['name']}: [{c['report']}]({c['report']})")
            elif step == "trends":
                st.write("**Industry Trends:**", state["industry_trends"])
            elif step == "usecases":
                st.json(state["use_cases"])
            elif step == "resources":
                for ds in state["datasets"]:
                    st.markdown(f"- [{ds['name']}]({ds['url']}) | {ds['license']} | {ds['relevance']}")
            elif step == "evaluate":
                for uc in state["use_cases"]:
                    st.write(f"{uc['title']} - Priority: {uc['priority_score']}")
            elif step == "final":
                st.markdown(state["final_report"])
    return state

if st.button("Generate AI Strategy"):
    progress_area = st.container()
    result = generate_use_cases(company, progress_area)

    # Save report
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/{company}_ai_strategy_report.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(result["final_report"])

    # Download button
    with open(filename, "r", encoding="utf-8") as f:
        st.download_button("⬇️ Download Report", f, file_name=f"{company}_ai_strategy_report.md")
