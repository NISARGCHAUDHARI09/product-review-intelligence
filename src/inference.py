import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentPredictor:
    """
    Sentiment predictor using the fine-tuned DistilBERT model.

    The model is loaded entirely from the local model directory.
    No internet connection or Hugging Face download is required.

    Expected model structure:

        models/
        └── final_distilbert_model/
            ├── config.json
            ├── model.safetensors
            ├── tokenizer.json
            └── tokenizer_config.json
    """

    def __init__(self, model_path, max_length=128):

        self.model_path = str(model_path)
        self.max_length = max_length

        # ----------------------------------------------------
        # Select device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        # ----------------------------------------------------
        # Load tokenizer locally
        # ----------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        # ----------------------------------------------------
        # Load fine-tuned model locally
        # ----------------------------------------------------

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True
        )

        # ----------------------------------------------------
        # Move model to selected device
        # ----------------------------------------------------

        self.model.to(self.device)

        # ----------------------------------------------------
        # Evaluation mode
        # ----------------------------------------------------

        self.model.eval()

        # ----------------------------------------------------
        # Load label mapping
        # ----------------------------------------------------

        self.id2label = self.model.config.id2label

        # Safety fallback in case config does not contain labels
        if not self.id2label:

            self.id2label = {
                0: "NEGATIVE",
                1: "NEUTRAL",
                2: "POSITIVE"
            }


    # ========================================================
    # SINGLE TEXT PREDICTION
    # ========================================================

    def predict(self, text):
        """
        Predict sentiment for a single product review.

        Parameters
        ----------
        text : str
            Product review text.

        Returns
        -------
        dict
            {
                "label": predicted sentiment,
                "confidence": confidence score,
                "probabilities": probability for each class
            }
        """

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if not isinstance(text, str) or not text.strip():

            raise ValueError(
                "Review text cannot be empty."
            )

        text = text.strip()

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )

        # ----------------------------------------------------
        # Move tensors to device
        # ----------------------------------------------------

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = self.model(**inputs)

            probabilities = F.softmax(
                outputs.logits,
                dim=-1
            )

        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        predicted_class = torch.argmax(
            probabilities,
            dim=-1
        ).item()

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = probabilities[
            0,
            predicted_class
        ].item()

        # ----------------------------------------------------
        # Predicted label
        # ----------------------------------------------------

        label = self.id2label[predicted_class]

        # ----------------------------------------------------
        # Probability for every class
        # ----------------------------------------------------

        probability_dict = {
            self.id2label[i]: float(probabilities[0, i])
            for i in range(probabilities.shape[-1])
        }

        # ----------------------------------------------------
        # Return results
        # ----------------------------------------------------

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probability_dict
        }


    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    def predict_batch(self, texts):
        """
        Predict sentiment for multiple reviews.

        Parameters
        ----------
        texts : list[str]
            List of product reviews.

        Returns
        -------
        list[dict]
            Prediction results for each review.
        """

        if not texts:

            return []

        # ----------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------

        for text in texts:

            if not isinstance(text, str) or not text.strip():

                raise ValueError(
                    "All reviews must be non-empty strings."
                )

        cleaned_texts = [
            text.strip()
            for text in texts
        ]

        # ----------------------------------------------------
        # Tokenize batch
        # ----------------------------------------------------

        inputs = self.tokenizer(
            cleaned_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )

        # ----------------------------------------------------
        # Move tensors to device
        # ----------------------------------------------------

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Batch inference
        # ----------------------------------------------------

        with torch.inference_mode():

            outputs = self.model(**inputs)

            probabilities = F.softmax(
                outputs.logits,
                dim=-1
            )

        # ----------------------------------------------------
        # Generate results
        # ----------------------------------------------------

        results = []

        for i in range(len(cleaned_texts)):

            predicted_class = torch.argmax(
                probabilities[i]
            ).item()

            confidence = probabilities[
                i,
                predicted_class
            ].item()

            label = self.id2label[
                predicted_class
            ]

            probability_dict = {
                self.id2label[j]: float(
                    probabilities[i, j]
                )
                for j in range(
                    probabilities.shape[-1]
                )
            }

            results.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "probabilities": probability_dict
                }
            )

        return results