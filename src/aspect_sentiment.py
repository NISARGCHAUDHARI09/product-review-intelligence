import re
from typing import List, Dict


class AspectSentimentAnalyzer:
    """
    Aspect-Based Sentiment Analysis using the fine-tuned
    DistilBERT sentiment classifier.

    The analyzer:
    1. Detects product-related aspects.
    2. Extracts the most relevant clause containing each aspect.
    3. Classifies that clause using the fine-tuned model.
    """

    def __init__(self, predictor):

        self.predictor = predictor

        # Keep longer/more specific aspects first.
        self.aspect_keywords = [
            "battery life",
            "camera quality",
            "build quality",
            "customer service",

            "battery",
            "camera",
            "screen",
            "display",
            "sound",
            "audio",
            "speaker",
            "speakers",
            "quality",
            "price",
            "cost",
            "value",
            "performance",
            "design",
            "build",
            "size",
            "weight",
            "taste",
            "flavor",
            "packaging",
            "delivery",
            "shipping",
            "durability",
            "material",
            "comfort",
            "service",
            "smell",
            "texture"
        ]


    def extract_aspects(self, text: str) -> List[str]:
        """
        Extract product aspects while avoiding
        overlapping duplicate aspects.
        """

        text_lower = text.lower()

        found = []

        # Search longer phrases first.
        for aspect in self.aspect_keywords:

            pattern = r"\b" + re.escape(aspect) + r"\b"

            if re.search(pattern, text_lower):

                # Don't add a shorter aspect if it is
                # already covered by a longer aspect.
                already_covered = any(
                    aspect in existing
                    for existing in found
                )

                if not already_covered:
                    found.append(aspect)

        return found


    def get_aspect_sentence(self, review: str, aspect: str) -> str:
        """
        Extract the smallest useful text span around an aspect.

        The method:
        1. Normalizes whitespace.
        2. Splits the review into sentences.
        3. Finds the sentence containing the aspect.
        4. Splits that sentence into clauses using common
           conjunctions such as 'and', 'but', 'while', etc.
        5. Returns the clause containing the aspect.

        This helps prevent unrelated sentiment words from
        influencing the aspect-level sentiment prediction.
        """

        # Normalize whitespace.
        text = re.sub(r"\s+", " ", review).strip()

        # Split into sentences.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        aspect_lower = aspect.lower()

        for sentence in sentences:

            if aspect_lower not in sentence.lower():
                continue

            # Split into smaller clauses.
            clauses = re.split(
                r"\s+(?:and|but|while|although|however)\s+",
                sentence,
                flags=re.IGNORECASE
            )

            # Return the clause containing the aspect.
            for clause in clauses:

                if aspect_lower in clause.lower():
                    return clause.strip()

            # Fallback to the complete sentence.
            return sentence.strip()

        # If the aspect cannot be located,
        # return the complete review as a safe fallback.
        return text


    def analyze(self, text: str) -> Dict:
        """
        Perform aspect-based sentiment analysis.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Review text cannot be empty."
            )

        aspects = self.extract_aspects(text)

        results = []

        for aspect in aspects:

            # Get text specifically related to this aspect.
            context = self.get_aspect_sentence(
                text,
                aspect
            )

            # Predict sentiment using the fine-tuned
            # DistilBERT model.
            prediction = self.predictor.predict(
                context
            )

            results.append(
                {
                    "aspect": aspect,
                    "sentiment": prediction["label"],
                    "confidence": prediction["confidence"],
                    "context": context
                }
            )

        return {
            "review": text,
            "aspects": results
        }