import streamlit as st
import json
from typing import TypedDict
from groq import Groq
from tavily import TavilyClient
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from pypdf import PdfReader

GROQ_API_KEY = "removed_for_security"
TAVILY_API_KEY = "removed_for_security"

llm = Groq(api_key=GROQ_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

ORCHESTRATOR_PROMPT = """
You are an intelligent ORCHESTRATOR controlling a multi-agent Learning Path Generator.

Your role:
- Observe the CURRENT STATE of the learning journey
- Decide which ONE agent should run NEXT
- You are allowed to RE-RUN agents if needed for quality improvement
- You are allowed to STOP when the learning roadmap is complete and high-quality

Available agents and their purpose:
- MCQ_AGENT → Generate assessment questions to evaluate learner's knowledge
- EVALUATE_AGENT → Score the learner's answers and calculate performance
- SKILL_GAP_AGENT → Analyze weak areas and identify learning gaps
- RESEARCH_AGENT → Search for high-quality learning resources online
- ROADMAP_AGENT → Build a personalized 4-week learning roadmap
- REFLECT_AGENT → Perform quality check on the roadmap (self-improvement)
- EXPLAIN_AGENT → Generate transparent explanation of AI decisions
- DONE → Finish the process and present results to user

CRITICAL STOPPING RULES - You MUST choose DONE when:
1. phase is "awaiting_answers" (MCQs generated, waiting for user) → DONE
2. phase is "complete" (everything finished) → DONE
3. has_mcqs=true AND has_answers=false → DONE (waiting for user input)
4. All of these are true: has_answers=true, has_roadmap=true, has_explanation=true, reflection_result="GOOD" → DONE

Workflow logic:
- If phase="mcq_generation" and has_mcqs=false → MCQ_AGENT
- If has_mcqs=true and has_answers=false → DONE (wait for user)
- If has_answers=true and score=null → EVALUATE_AGENT
- If score exists and has_skill_gaps=false → SKILL_GAP_AGENT
- If has_skill_gaps=true and has_roadmap=false → RESEARCH_AGENT
- If has_resources=true and has_roadmap=false → ROADMAP_AGENT
- If has_roadmap=true and has_reflection=false → REFLECT_AGENT
- If reflection_result="NEEDS_IMPROVEMENT" and retries<2 → ROADMAP_AGENT
- If reflection_result="GOOD" and has_explanation=false → EXPLAIN_AGENT
- Otherwise → DONE

Input:
You will receive a JSON object describing the CURRENT STATE.

Output rules:
- Choose EXACTLY ONE next action
- Return ONLY valid JSON with no markdown formatting
- No explanations outside JSON

Output format:
{
  "next_action": "MCQ_AGENT | EVALUATE_AGENT | SKILL_GAP_AGENT | RESEARCH_AGENT | ROADMAP_AGENT | REFLECT_AGENT | EXPLAIN_AGENT | DONE",
  "reason": "brief explanation of why this agent should run next"
}
"""

def fallback_mcqs(goal: str):
    return [
        {"q": f"What is the primary purpose of {goal}?", "o": ["Option A", "Option B", "Option C", "Option D"], "a": "Option A"},
        {"q": f"Which concept is fundamental to {goal}?", "o": ["Option A", "Option B", "Option C", "Option D"], "a": "Option B"},
        {"q": f"What is a key application of {goal}?", "o": ["Option A", "Option B", "Option C", "Option D"], "a": "Option C"},
        {"q": f"What challenge exists in {goal}?", "o": ["Option A", "Option B", "Option C", "Option D"], "a": "Option D"},
        {"q": f"What is the future scope of {goal}?", "o": ["Option A", "Option B", "Option C", "Option D"], "a": "Option A"},
    ]

class AgentState(TypedDict):
    phase: str
    goal: str
    mcqs: list
    answers: list
    score: int
    skill_gaps: str
    roadmap: str
    resources: str
    reflection: str
    explanation: str
    retries: int
    roadmap_built: bool


@tool
def generate_mcqs(goal: str) -> list:
    """Generate MCQs with fallback."""
    prompt = f"""
    You are an expert educator. Generate 5 multiple-choice questions about: {goal}
    
    Output ONLY valid JSON in this exact format:
    {{
      "questions": [
        {{"q": "Question text here?", "o": ["Option A", "Option B", "Option C", "Option D"], "a": "Option A"}},
        {{"q": "Question text here?", "o": ["Option A", "Option B", "Option C", "Option D"], "a": "Option B"}}
      ]
    }}
    
    Make questions comprehensive covering basics, applications, and challenges.
    """
    try:
        res = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = res.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        return parsed.get("questions", fallback_mcqs(goal))
    except Exception as e:
        print(f"MCQ generation error: {e}")
        return fallback_mcqs(goal)

@tool
def research_resources(goal: str, level: str) -> str:
    """Research learning resources using Tavily."""
    try:
        query = f"{goal} {level} tutorial learning resources best practices"
        results = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )
        formatted = "\n".join([
            f"- {r.get('title', 'Resource')}: {r.get('url', '')}" 
            for r in results.get('results', [])
        ])
        return formatted or "General online tutorials and documentation recommended."
    except Exception as e:
        print(f"Research error: {e}")
        return "General online tutorials and documentation recommended."

@tool
def build_roadmap(goal: str, gaps: str, resources: str) -> str:
    """Build personalized learning roadmap."""
    prompt = f"""
    You are an expert learning coach. Create a detailed 4-week learning roadmap.
    
    Goal: {goal}
    Weak Areas: {gaps}
    Available Resources: {resources}
    
    Structure the roadmap as:
    
    **Week 1: Foundation**
    - Day 1-2: [specific topics]
    - Day 3-4: [specific topics]
    - Day 5-7: [practice exercises]
    
    **Week 2: Core Concepts**
    [similar structure]
    
    **Week 3: Advanced Topics**
    [similar structure]
    
    **Week 4: Practice & Projects**
    [similar structure]
    
    Include specific learning objectives, resources, and practice exercises.
    """
    try:
        res = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Error generating roadmap: {e}"

@tool
def reflect_and_improve(roadmap: str) -> str:
    """Self-reflection on roadmap quality."""
    prompt = f"""
    You are a quality control expert. Review this learning roadmap:
    
    {roadmap}
    
    Check if it:
    1. Has clear weekly structure
    2. Includes specific topics and milestones
    3. Has practical exercises
    4. Is realistic and actionable
    
    If ALL criteria are met, respond with ONLY: "GOOD_ENOUGH"
    
    If improvements needed, respond with: "NEEDS_IMPROVEMENT: [specific suggestions]"
    """
    try:
        res = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return res.choices[0].message.content
    except Exception as e:
        return "GOOD_ENOUGH"

@tool
def explain_ai_decisions(score: int, skill_gaps: str, roadmap: str, retries: int) -> str:
    """Explain AI decision-making process."""
    prompt = f"""
    Explain the AI's decision-making process in simple terms:
    
    - Assessment Score: {score}/5
    - Identified Gaps: {skill_gaps}
    - Roadmap Generated: {'Yes' if roadmap else 'No'}
    - Quality Checks: {retries} iterations
    
    Provide a brief, friendly explanation of how the AI analyzed the user's performance 
    and created a personalized learning path.
    """
    try:
        res = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return res.choices[0].message.content
    except Exception as e:
        return "The AI analyzed your responses and created a personalized learning roadmap based on your current knowledge level."

@tool
def mentor_chat(question: str, roadmap: str) -> str:
    """Intelligent mentor chatbot."""
    prompt = f"""
    You are a helpful learning mentor. Here is the student's roadmap:
    
    {roadmap}
    
    Student Question: {question}
    
    Provide a helpful, encouraging response that:
    - Addresses their specific question
    - References relevant parts of their roadmap
    - Offers practical next steps
    - Stays positive and motivating
    """
    try:
        res = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content
    except Exception as e:
        return "I'm here to help! Could you rephrase your question?"


def evaluate_answers(mcqs: list, answers: list) -> int:
    """Evaluate answers safely."""
    if not answers or not mcqs:
        return 0
    score = 0
    for i, (q, a) in enumerate(zip(mcqs, answers)):
        if i < len(answers) and a == q.get("a"):
            score += 1
    return score

def detect_skill_gaps(mcqs: list, answers: list) -> str:
    """Detect weak concepts safely."""
    if not answers or not mcqs:
        return "Assessment not completed yet."
    wrong = []
    for i, (q, a) in enumerate(zip(mcqs, answers)):
        if i < len(answers) and a != q.get("a"):
            wrong.append(q.get("q", f"Question {i+1}"))
    
    if not wrong:
        return "Strong understanding across all areas! Let's build on this foundation."
    return "Areas for improvement:\n" + "\n".join(f"- {w}" for w in wrong)

def llm_orchestrator(state: AgentState) -> str:
    """
    LLM-powered orchestrator that autonomously decides the next agent.
    This is TRUE agentic AI - the LLM thinks and plans dynamically.
    """
    # CRITICAL: Check if we should stop before asking LLM
    phase = state.get("phase", "mcq_generation")
    
    # Stop if waiting for user input
    if phase == "awaiting_answers":
        print("🛑 STOPPING: Waiting for user to answer MCQs")
        return "DONE"
    
    # Stop if complete
    if phase == "complete":
        print("🛑 STOPPING: Process complete")
        return "DONE"
    
    # Stop if MCQs exist but no answers yet (waiting for user)
    if state.get("mcqs") and not state.get("answers"):
        print("🛑 STOPPING: MCQs ready, waiting for user answers")
        return "DONE"
    
    state_summary = {
        "phase": phase,
        "goal": state.get("goal", "")[:100],  
        "has_mcqs": bool(state.get("mcqs")),
        "has_answers": bool(state.get("answers")),
        "has_resources": bool(state.get("resources")),
        "score": state.get("score"),
        "has_skill_gaps": bool(state.get("skill_gaps")),
        "has_roadmap": bool(state.get("roadmap")),
        "has_reflection": bool(state.get("reflection")),
        "reflection_result": "GOOD" if state.get("reflection") and "GOOD_ENOUGH" in state.get("reflection", "") else "NEEDS_IMPROVEMENT" if state.get("reflection") else "NOT_DONE",
        "has_explanation": bool(state.get("explanation")),
        "retries": state.get("retries", 0)
    }
    
    print(f"\n🧠 LLM ORCHESTRATOR THINKING...")
    print(f"   State: {json.dumps(state_summary, indent=2)}")
    
    try:
        response = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": ORCHESTRATOR_PROMPT},
                {"role": "user", "content": f"Current state:\n{json.dumps(state_summary, indent=2)}"}
            ],
            temperature=0.3
        )
        
        decision_text = response.choices[0].message.content.strip()
        
        if decision_text.startswith("```"):
            decision_text = decision_text.split("```")[1]
            if decision_text.startswith("json"):
                decision_text = decision_text[4:]
        
        decision = json.loads(decision_text)
        
        next_action = decision.get("next_action", "DONE")
        reason = decision.get("reason", "No reason provided")
        
        print(f"   🎯 Decision: {next_action}")
        print(f"   💭 Reason: {reason}\n")
        
        return next_action
        
    except Exception as e:
        print(f"   ❌ Orchestrator error: {e}")
        print(f"   → Falling back to DONE\n")
        return "DONE"

def mcq_node(s):
    """MCQ generation agent."""
    print("🤖 MCQ_AGENT: Generating assessment questions...")
    mcqs = generate_mcqs.invoke(input={"goal": s["goal"]})
    return {"mcqs": mcqs, "phase": "awaiting_answers"}

def eval_node(s):
    """Evaluation agent."""
    print("🤖 EVALUATE_AGENT: Scoring answers...")
    score = evaluate_answers(s["mcqs"], s["answers"])
    print(f"   Score: {score}/5")
    return {"score": score}

def gap_node(s):
    """Skill gap analysis agent."""
    print("🤖 SKILL_GAP_AGENT: Analyzing weak areas...")
    gaps = detect_skill_gaps(s["mcqs"], s["answers"])
    return {"skill_gaps": gaps}

def research_node(s):
    """Research agent."""
    print("🤖 RESEARCH_AGENT: Finding learning resources...")
    level = "Beginner" if s["score"] < 3 else "Intermediate" if s["score"] < 4 else "Advanced"
    resources = research_resources.invoke(input={"goal": s["goal"], "level": level})
    return {"resources": resources}

def roadmap_node(s):
    """Roadmap building agent."""
    print("🤖 ROADMAP_AGENT: Building personalized learning path...")
    
    resources = s.get("resources", "")
    if not resources:
        level = "Beginner" if s["score"] < 3 else "Intermediate" if s["score"] < 4 else "Advanced"
        resources = research_resources.invoke(input={"goal": s["goal"], "level": level})
    
    roadmap = build_roadmap.invoke(input={
        "goal": s["goal"],
        "gaps": s["skill_gaps"],
        "resources": resources
    })
    
    return {
        "roadmap": roadmap,
        "resources": resources,
        "roadmap_built": True,
        "reflection": ""  
    }

def reflect_node(s):
    """Self-reflection agent."""
    print("🤖 REFLECT_AGENT: Performing quality check...")
    reflection = reflect_and_improve.invoke(input={"roadmap": s["roadmap"]})
    is_good = "GOOD_ENOUGH" in reflection
    
    new_retries = s["retries"] + (0 if is_good else 1)
    
    print(f"   Quality: {'✅ PASSED' if is_good else '⚠️ NEEDS IMPROVEMENT'}")
    print(f"   Retries: {new_retries}")
    
    return {
        "reflection": reflection,
        "retries": new_retries,
        "roadmap": "" if not is_good and new_retries < 2 else s["roadmap"],
        "roadmap_built": is_good
    }

def explain_node(s):
    """Explainable AI agent."""
    print("🤖 EXPLAIN_AGENT: Generating transparency report...")
    explanation = explain_ai_decisions.invoke(input={
        "score": s.get("score", 0),
        "skill_gaps": s.get("skill_gaps", "None"),
        "roadmap": s.get("roadmap", ""),
        "retries": s.get("retries", 0)
    })
    return {
        "explanation": explanation,
        "phase": "complete"
    }

def route_based_on_llm(state: AgentState):
    """Route to next node based on LLM orchestrator decision."""
    decision = llm_orchestrator(state)
    
    routing_map = {
        "MCQ_AGENT": "MCQ",
        "EVALUATE_AGENT": "EVALUATE",
        "SKILL_GAP_AGENT": "SKILL_GAP",
        "RESEARCH_AGENT": "RESEARCH",
        "ROADMAP_AGENT": "ROADMAP",
        "REFLECT_AGENT": "REFLECT",
        "EXPLAIN_AGENT": "EXPLAIN",
        "DONE": "END"
    }
    
    return routing_map.get(decision, "END")

graph = StateGraph(AgentState)

graph.add_node("ORCHESTRATOR", lambda s: s)

graph.add_node("MCQ", mcq_node)
graph.add_node("EVALUATE", eval_node)
graph.add_node("SKILL_GAP", gap_node)
graph.add_node("RESEARCH", research_node)
graph.add_node("ROADMAP", roadmap_node)
graph.add_node("REFLECT", reflect_node)
graph.add_node("EXPLAIN", explain_node)

graph.set_entry_point("ORCHESTRATOR")

graph.add_conditional_edges(
    "ORCHESTRATOR",
    route_based_on_llm,
    {
        "MCQ": "MCQ",
        "EVALUATE": "EVALUATE",
        "SKILL_GAP": "SKILL_GAP",
        "RESEARCH": "RESEARCH",
        "ROADMAP": "ROADMAP",
        "REFLECT": "REFLECT",
        "EXPLAIN": "EXPLAIN",
        "END": END
    }
)

for node in ["MCQ", "EVALUATE", "SKILL_GAP", "RESEARCH", "ROADMAP", "REFLECT", "EXPLAIN"]:
    graph.add_edge(node, "ORCHESTRATOR")

app = graph.compile()

st.set_page_config(page_title="MargDarshan", layout="wide", page_icon="🧠")

st.title("🧠 MargDarshan")
st.markdown("AI-Powered Personalized Learning Path Generator.")

if "state" not in st.session_state:
    st.session_state.state = {
        "phase": "mcq_generation",
        "goal": "",
        "mcqs": [],
        "answers": [],
        "score": None,
        "skill_gaps": "",
        "roadmap": "",
        "resources": "",
        "reflection": "",
        "explanation": "",
        "retries": 0,
        "roadmap_built": False
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📚 Step 1: Define Your Learning Goal")
    pdf = st.file_uploader("Upload Syllabus PDF (Optional)", type="pdf")
    topic = st.text_input("Or Enter Learning Topic", placeholder="e.g., Machine Learning, Python Programming, Data Structures")

with col2:
    st.subheader("🎯 Current Phase")
    phase_display = {
        "mcq_generation": "🔄 Generating Assessment",
        "awaiting_answers": "✏️ Ready for Assessment",
        "roadmap_generation": "🚀 Building Roadmap",
        "complete": "✅ Complete"
    }
    st.info(phase_display.get(st.session_state.state.get("phase", "mcq_generation"), "Ready"))

if st.button("🚀 Start Assessment", type="primary", use_container_width=True):
    if pdf or topic:
        with st.spinner("🤖 LLM Orchestrator is thinking and coordinating agents..."):
            if pdf:
                reader = PdfReader(pdf)
                goal_text = " ".join(p.extract_text() for p in reader.pages[:3])
                st.session_state.state["goal"] = goal_text[:500]
            else:
                st.session_state.state["goal"] = topic
            
            st.session_state.state.update({
                "phase": "mcq_generation",
                "mcqs": [],
                "answers": [],
                "score": None,
                "skill_gaps": "",
                "roadmap": "",
                "resources": "",
                "reflection": "",
                "explanation": "",
                "retries": 0,
                "roadmap_built": False
            })
            
            st.session_state.state = app.invoke(
                st.session_state.state,
                config={"recursion_limit": 100}
            )
        
        st.success("✅ Assessment ready! The LLM orchestrator generated your questions.")
        st.rerun()
    else:
        st.warning("⚠️ Please upload a PDF or enter a topic first.")

if st.session_state.state.get("mcqs") and st.session_state.state.get("phase") == "awaiting_answers":
    st.markdown("---")
    st.subheader("🧠 Knowledge Assessment")
    st.write("Answer these questions - the LLM will analyze your responses:")
    
    answers = []
    for i, q in enumerate(st.session_state.state["mcqs"]):
        st.markdown(f"**Question {i+1}:** {q['q']}")
        answer = st.radio(
            f"Select your answer:",
            options=q["o"],
            key=f"q_{i}",
            label_visibility="collapsed"
        )
        answers.append(answer)
    
    if st.button("📊 Submit Assessment", type="primary", use_container_width=True):
        with st.spinner("🤖 LLM Orchestrator is analyzing and building your roadmap..."):
            st.session_state.state["answers"] = answers
            st.session_state.state["phase"] = "roadmap_generation"
            
            st.session_state.state = app.invoke(
                st.session_state.state,
                config={"recursion_limit": 100}
            )
        
        st.success("✅ Your personalized roadmap is ready!")
        st.rerun()

if st.session_state.state.get("phase") == "complete":
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Score", f"{st.session_state.state.get('score', 0)}/5")
    with col2:
        score = st.session_state.state.get('score', 0)
        level = "Beginner" if score < 3 else "Intermediate" if score < 4 else "Advanced"
        st.metric("🎯 Level", level)
    with col3:
        st.metric("🔄 Quality Iterations", st.session_state.state.get('retries', 0) + 1)
    
    if st.session_state.state.get("skill_gaps"):
        st.subheader("📋 Skill Gap Analysis")
        st.info(st.session_state.state["skill_gaps"])
    
    if st.session_state.state.get("roadmap"):
        st.subheader("🗺️ Your Personalized Learning Roadmap")
        st.markdown(st.session_state.state["roadmap"])
        
        if st.session_state.state.get("resources"):
            with st.expander("📚 Recommended Resources"):
                st.markdown(st.session_state.state["resources"])
    
    if st.session_state.state.get("explanation"):
        with st.expander("🔍 How the LLM Orchestrator Made Decisions"):
            st.markdown(st.session_state.state["explanation"])

if st.session_state.state.get("roadmap"):
    st.markdown("---")
    st.subheader("💬 AI Mentor Chat")
    st.write("Ask questions about your roadmap:")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if user_question := st.chat_input("Ask your mentor..."):
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.spinner("🤔 Mentor is thinking..."):
            response = mentor_chat.invoke(input={
                "question": user_question,
                "roadmap": st.session_state.state["roadmap"]
            })
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()