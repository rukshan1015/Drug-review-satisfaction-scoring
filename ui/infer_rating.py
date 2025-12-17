from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import gradio as gr

MODEL_ID = "rukshan1015/drug-review-bert-regression-fullmodel"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
model.eval()


def predict_rating(text, max_length=128):
    """
    Predict a 1–10 satisfaction rating from a single review string
    or a list of review strings.
    """
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        preds = outputs.logits.squeeze(-1).cpu().numpy()

    if isinstance(text, str):
        return float(preds.item())
    return preds.tolist()


def gradio_predict(drug_name: str, condition: str, review_text: str) -> str:
    if not review_text.strip():
        return "Please enter a review."

    # same pattern used for training
    full_text = f"Drug: {drug_name}. Condition: {condition}. Review: {review_text}"

    score = predict_rating(full_text)
    return f"Predicted satisfaction rating: {score:.2f} (scale 1–10)"


# Building Gradio UI


demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Textbox(label="Drug name", placeholder="e.g., Valsartan"),
        gr.Textbox(label="Condition", placeholder="e.g., Hypertension"),
        gr.Textbox(
            lines=6,
            label="Review",
            placeholder='e.g., "It has no side effect, I take it in combination..."'
        ),
    ],
    outputs=gr.Textbox(label="Model prediction"),
    title="Drug Review Satisfaction Scorer",
    description="Predicts a 1–10 satisfaction rating using a fine-tuned DistilBERT model."
)


if __name__ == "__main__":
    # Launch local Gradio app
    #demo.launch()
    demo.launch(server_name="0.0.0.0", server_port=7860)
