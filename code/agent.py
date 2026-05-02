import json
from pydantic import BaseModel, Field
import ollama
from config import CHAT_MODEL
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
        self.retriever = Retriever()
        
    def process_ticket(self, issue: str, subject: str, company: str) -> dict:
        # 1. Retrieve Context
        query = f"Subject: {subject}\nIssue: {issue}"
        context = self.retriever.retrieve(query=query, company=company, top_k=3)
        
        # 2. Construct Prompt
        system_prompt = """You are an AI Support Agent. Classify the ticket status strictly as either 'replied' or 'escalated'.

CRITICAL INSTRUCTIONS:
1. ONLY set status to 'escalated' if the user asks for a refund, account deletion, mentions security/PII, is angry, OR if the provided context DOES NOT contain the answer.
2. If the context DOES contain the answer and it is a normal question, you MUST set status to 'replied'.
3. The 'response' and 'justification' fields MUST be simple PLAIN TEXT strings. Do NOT output nested JSON objects inside them.

Context Documents:
{context}
"""
        
        user_prompt = f"""Support Ticket Details:
Company: {company}
Subject: {subject}
Issue: {issue}

Respond strictly with a single valid JSON object matching the schema."""

        # 3. Call Ollama with Structured Output
        try:
            response = ollama.chat(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt.format(context=context)},
                    {"role": "user", "content": user_prompt}
                ],
                format=TicketResponse.model_json_schema(),
                options={"temperature": 0.0}
            )
            
            # The response text will be a valid JSON string matching the schema
            result_dict = json.loads(response['message']['content'])
            return result_dict
            
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            # Return a fallback escalated response
            return {
                "status": "escalated",
                "product_area": "unknown",
                "response": "An error occurred while processing the ticket. Escalating to human.",
                "justification": f"System error: {str(e)}",
                "request_type": "invalid"
            }
