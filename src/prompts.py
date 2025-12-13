"""
Prompts module using Pydantic for structured prompt management.

This module defines prompt templates with Pydantic validation for:
- RAG Q&A prompts
- Context grounding prompts
- Response validation prompts
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class PromptTemplate(BaseModel):
    """Base prompt template model."""
    name: str = Field(..., description="Unique name for the prompt")
    description: str = Field(..., description="Description of what the prompt does")
    template: str = Field(..., description="The actual prompt template with placeholders")
    variables: List[str] = Field(default_factory=list, description="List of variables used in template")

    def format(self, **kwargs) -> str:
        """Format the prompt template with provided variables."""
        return self.template.format(**kwargs)


class RAGPrompt(PromptTemplate):
    """RAG Q&A prompt template."""
    system_message: str = Field(
        default="You are a helpful assistant that answers questions based on provided context.",
        description="System message for the LLM"
    )


class ContextGroundedPrompt(PromptTemplate):
    """Prompt for ensuring response is grounded in context."""
    context_placeholder: str = Field(
        default="{context}",
        description="Placeholder for context in template"
    )
    question_placeholder: str = Field(
        default="{question}",
        description="Placeholder for question in template"
    )


class ResponseValidationPrompt(PromptTemplate):
    """Prompt for validating that response is grounded in context."""
    pass


# Pre-defined prompt templates
RAG_QA_PROMPT = RAGPrompt(
    name="rag_qa",
    description="Standard RAG question-answering prompt",
    template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
If the context includes image descriptions, consider those as visual information.

Context:
{context}

Question: {question}

Answer:""",
    variables=["context", "question"],
    system_message="You are a helpful assistant that answers questions based on provided context. Be concise and accurate."
)

CONTEXT_GROUNDED_PROMPT = ContextGroundedPrompt(
    name="context_grounded",
    description="Prompt ensuring response is grounded in provided context",
    template="""Based ONLY on the following context, answer the question. Do not use external knowledge.
If the answer is not in the context, say 'This information is not available in the provided context.'

Context:
{context}

Question: {question}

Answer:""",
    variables=["context", "question"],
    context_placeholder="{context}",
    question_placeholder="{question}"
)

VALIDATION_PROMPT = ResponseValidationPrompt(
    name="response_validation",
    description="Prompt for validating response is grounded in context",
    template="""You are a validator. Check if the following answer is grounded in the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Validation result (just respond with 'VALID' or 'INVALID'):""",
    variables=["context", "question", "answer"]
)

SUMMARIZATION_PROMPT = PromptTemplate(
    name="summarization",
    description="Prompt for summarizing retrieved context",
    template="""Summarize the following context in a concise manner, highlighting the key points relevant to the question:

Context:
{context}

Question: {question}

Summary:""",
    variables=["context", "question"]
)

CLARIFICATION_PROMPT = PromptTemplate(
    name="clarification",
    description="Prompt for clarifying ambiguous questions",
    template="""The user's question might be ambiguous. Based on the following context, provide clarifications or ask for more specifics if needed.

Context:
{context}

Question: {question}

Clarification:""",
    variables=["context", "question"]
)


class PromptLibrary:
    """Library to manage and retrieve prompt templates."""

    def __init__(self):
        """Initialize prompt library with pre-defined prompts."""
        self.prompts = {
            "rag_qa": RAG_QA_PROMPT,
            "context_grounded": CONTEXT_GROUNDED_PROMPT,
            "response_validation": VALIDATION_PROMPT,
            "summarization": SUMMARIZATION_PROMPT,
            "clarification": CLARIFICATION_PROMPT,
        }

    def get_prompt(self, name: str) -> Optional[PromptTemplate]:
        """Retrieve a prompt template by name."""
        return self.prompts.get(name)

    def list_prompts(self) -> List[str]:
        """List all available prompt names."""
        return list(self.prompts.keys())

    def add_prompt(self, prompt: PromptTemplate) -> None:
        """Add a new prompt template to the library."""
        self.prompts[prompt.name] = prompt

    def format_prompt(self, name: str, **kwargs) -> str:
        """Format a prompt by name with provided variables."""
        prompt = self.get_prompt(name)
        if not prompt:
            raise ValueError(f"Prompt '{name}' not found in library")
        return prompt.format(**kwargs)


# Global prompt library instance
prompt_library = PromptLibrary()
