
import logging
import os
import streamlit as st

from rag_backend import rag_query

logger = logging.getLogger("lalimsalita.app")


def get_groq_api_key():
    # Render environment variables take priority.
    key = os.getenv("GROQ_API_KEY")

    if key:
        return key

    # Local Streamlit secrets are also supported.
    try:
        return st.secrets.get("GROQ_API_KEY")
    except (FileNotFoundError, KeyError):
        return None


def generate_chatbot_response(question):

    try:
        api_key = get_groq_api_key()

        if not api_key:
            logger.error(
                "Missing GROQ_API_KEY environment variable"
            )
            return {
                "answer": (
                    "Hindi pansamantalang available ang "
                    "serbisyo ng chatbot. Pakisubukang muli."
                ),
                "retrieved": []
            }

        result = rag_query(
            question,
            api_key=api_key
        )

        return {
            "answer": result["answer"],
            "retrieved": result.get("retrieved", [])
        }

    except Exception:
        logger.exception(
            "LalimSalita request failed"
        )

        return {
            "answer": (
                "May problema sa pagkuha ng sagot. "
                "Pakisubukang muli."
            ),
            "retrieved": []
        }
