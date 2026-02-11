import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END

# --- 1. BERT SETUP ---
MODEL_ID = "rukshan1015/drug-review-bert-regression-fullmodel"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
model.eval()

# Agentic fokflow - we will have two nodes: 1) BERT Specialist and 2) Clinical Auditor. The state will be passed between them.
class AgentState(TypedDict):
    """The state passed between nodes in the graph."""
    full_text: str
    score: float
    audit_report: str
    needs_human_review: bool

def bert_scoring_node(state: AgentState) -> Dict:
    """Node 1: Use the fine-tuned BERT model to get the numerical score."""
    inputs = tokenizer(state['full_text'], padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        score = int(outputs.logits.squeeze(-1).cpu().numpy().item())
    
    return {"score": score}

def clinical_audit_node(state: AgentState) -> Dict:
    """Node 2: Logic-based Auditor that decides if the review is a safety risk."""
    score = state['score']
    if score < 5.0:
        report = f"🚨 ALERT: Low satisfaction score ({score:.2f}). Triggering Adverse Event extraction..."
        review_flag = True
    else:
        report = f"✅ PASS: Satisfaction score ({score:.2f}) is within normal parameters."
        review_flag = False
    
    return {"audit_report": report, "needs_human_review": review_flag}


workflow = StateGraph(AgentState)


workflow.add_node("bert_specialist", bert_scoring_node)
workflow.add_node("clinical_auditor", clinical_audit_node)


workflow.set_entry_point("bert_specialist")
workflow.add_edge("bert_specialist", "clinical_auditor")
workflow.add_edge("clinical_auditor", END)

audit_app = workflow.compile()

# Gradio Interface
def run_agentic_audit(drug_name, condition, review_text):
    if not review_text.strip():
        return "N/A", "Please enter a review."

    input_text = f"Drug: {drug_name}. Condition: {condition}. Review: {review_text}"
    
    initial_state = {"full_text": input_text, "score": 0.0, "audit_report": "", "needs_human_review": False}
    result = audit_app.invoke(initial_state)
    
    return f"{result['score']:.2f} / 10.0", result['audit_report']

with gr.Blocks(title="Pfizer IAS Demo") as demo:
    gr.Markdown("# 🏥 LangGraph Clinical Safety Auditor")
    gr.Markdown("This system uses **LangGraph** to orchestrate a fine-tuned **BERT specialist** and an **Automated Auditor**.")
    
    with gr.Row():
        with gr.Column():
            drug = gr.Textbox(label="Drug Name")
            cond = gr.Textbox(label="Condition")
            rev = gr.Textbox(lines=5, label="Patient Review")
            btn = gr.Button("Execute Agentic Workflow", variant="primary")
        with gr.Column():
            out_score = gr.Label(label="BERT Node Output")
            out_report = gr.Textbox(label="Auditor Node Output", lines=5)

    btn.click(fn=run_agentic_audit, inputs=[drug, cond, rev], outputs=[out_score, out_report])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)