import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentPredictor:
    def __init__(self, model_path="nisarg0902/product-review-distilbert-sentiment"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Review text cannot be empty.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)

        predicted_class = torch.argmax(
            probabilities,
            dim=-1
        ).item()

        confidence = probabilities[0][predicted_class].item()

        label = self.model.config.id2label[predicted_class]

        probability_dict = {
            self.model.config.id2label[i]: float(probabilities[0][i])
            for i in range(len(probabilities[0]))
        }

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probability_dict
        }