import json
from pydantic import BaseModel, Field
from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL
from retriever import Retriever
from typing import Literal

# Pydantic schema for structured output
class TicketResponse(BaseModel):
    status: Literal["replied", "escalated"] = Field(description="Whether the agent should answer directly or escalate")
    product_area: str = Field(description="The most relevant support category or domain area")
    response: str = Field(description="A user-facing answer grounded in the support corpus. If escalating, keep it brief and state it is being escalated.")
    justification: str = Field(description="A concise explanation of the decision & response, trace to the corpus")
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"] = Field(description="The best-fit request classification")

class Agent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.retriever = Retriever()
        
    def process_ticket(self, issue: str, subject: str, company: str) -> dict:
        if not self.client:
            raise ValueError("OPENAI_API_KEY is not set.")
            
        # 1. Retrieve Context
        query = f"Subject: {subject}\nIssue: {issue}"
        context = self.retriever.retrieve(query=query, company=company, top_k=3)
        
        # 2. Construct Prompt
        system_prompt = """You are a terminal-based support triage agent handling tickets for HackerRank, Claude, and Visa.
        
You MUST use ONLY the provided context corpus to understand the issue, decide whether it can be answered safely, and determine when it should be escalated. Do NOT use outside knowledge.

RULES:
- assess urgency and risk. If high-risk, sensitive, or unsupported, escalate.
- identify the request type and product area.
- generate a safe, grounded response if replying.
- if escalating, the response should be brief (e.g., stating the issue is being escalated to a human).
- NO hallucinated policies or steps.

Context Corpus:
{context}
"""
        
        user_prompt = f"""Support Ticket Details:
Company: {company}
Subject: {subject}
Issue: {issue}

Generate the response following the exact JSON schema."""

        # 3. Call OpenAI with Structured Output
        try:
            completion = self.client.beta.chat.completions.parse(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt.format(context=context)},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=TicketResponse,
                temperature=0.0
            )
            
            result = completion.choices[0].message.parsed
            return result.model_dump()
            
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            # Return a fallback escalated response
            return {
                "status": "escalated",
                "product_area": "unknown",
                "response": "An error occurred while processing the ticket. Escalating to human.",
                "justification": f"System error: {str(e)}",
                "request_type": "invalid"
            }
