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


def gradio_predict(review_text: str) -> str:
   
    if not review_text.strip():
        return "Please enter a review."

    score = predict_rating(review_text)
    return f"Predicted satisfaction rating: {score:.2f} (scale 1–10)"


# Build Gradio UI
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(
        lines=6,
        label="Enter drug review",
        placeholder='Example: "It has no side effect, I take it in combination with Bystolic 5 mg and Fish Oil."'
    ),
    outputs=gr.Textbox(
        label="Model prediction"
    ),
    title="Drug Review Satisfaction Scorer",
    description=(
        "Fine-tuned DistilBERT model that predicts a 1–10 satisfaction rating "
        "from free-text patient drug reviews."
    )
)


if __name__ == "__main__":
    # Launch local Gradio app
    demo.launch()
