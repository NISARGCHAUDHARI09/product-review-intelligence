import re
from typing import Optional


class ReviewPreprocessor:
    """
    Reusable preprocessing utility for product reviews.

    This module performs lightweight text normalization while
    preserving sentiment-bearing information.

    Important:
    We do NOT remove words such as:
        - not
        - no
        - never
        - bad
        - excellent
        - terrible

    because these words can strongly affect sentiment.
    """

    def __init__(self):
        pass

    # ========================================================
    # REMOVE HTML
    # ========================================================

    @staticmethod
    def remove_html(text: str) -> str:
        """
        Remove HTML tags from review text.

        Example:
            "<br />Great product"
            -> "Great product"
        """

        if not isinstance(text, str):
            return ""

        text = re.sub(
            r"<[^>]+>",
            " ",
            text
        )

        return text


    # ========================================================
    # NORMALIZE WHITESPACE
    # ========================================================

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Replace multiple spaces, tabs and newlines
        with a single space.
        """

        if not isinstance(text, str):
            return ""

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()


    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    def clean_text(self, text: str) -> str:
        """
        Perform lightweight review text cleaning.

        The function intentionally preserves:
            - punctuation
            - sentiment words
            - negations
            - numbers
            - capitalization information

        This is suitable for Transformer-based models.
        """

        if not isinstance(text, str):
            return ""

        # Remove HTML
        text = self.remove_html(text)

        # Normalize whitespace
        text = self.normalize_whitespace(text)

        return text


    # ========================================================
    # VALIDATE TEXT
    # ========================================================

    @staticmethod
    def is_valid_review(
        text: str,
        min_length: int = 3
    ) -> bool:
        """
        Check whether a review contains usable text.
        """

        if not isinstance(text, str):
            return False

        text = text.strip()

        if len(text) < min_length:
            return False

        return True


    # ========================================================
    # PROCESS REVIEW
    # ========================================================

    def process(
        self,
        text: str,
        validate: bool = True
    ) -> str:
        """
        Complete preprocessing pipeline for one review.

        Parameters
        ----------
        text : str
            Raw product review.

        validate : bool
            Whether to validate the cleaned review.

        Returns
        -------
        str
            Cleaned review text.

        Raises
        ------
        ValueError
            If the review is empty or invalid.
        """

        if not isinstance(text, str):
            raise ValueError(
                "Review must be a string."
            )

        cleaned_text = self.clean_text(text)

        if validate and not self.is_valid_review(
            cleaned_text
        ):
            raise ValueError(
                "Review text is empty or too short."
            )

        return cleaned_text


    # ========================================================
    # PROCESS OPTIONAL TEXT
    # ========================================================

    def process_optional(
        self,
        text: Optional[str]
    ) -> str:
        """
        Safely process optional review text.

        Returns an empty string instead of raising an error
        for missing values.
        """

        if text is None:
            return ""

        if not isinstance(text, str):
            return ""

        return self.clean_text(text)