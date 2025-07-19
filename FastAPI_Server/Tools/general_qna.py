import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage,BaseMessage
from langchain.tools import tool
from typing import List, Union

load_dotenv(override=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY")
)

@tool
def gen_qna(question: str, dependency_context: str = "",
    message_history: List[Union[AIMessage, HumanMessage]] = [],) -> str:
    """
    This tool is used for answering general questions.

    Parameters:
    - question: the user's query
    - context: (optional) relevant prior info from chat history, dependencies, or memories
    """
    
    def format_history(history: List[BaseMessage]) -> str:
            """Format message history into a string for model input."""
            formatted = ""
            for msg in history[-10:]:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                formatted += f"{role}: {msg.content}\n"
            return formatted.strip()

    history_str = format_history(message_history)
        # Step 1: Use LLM to intelligently formulate the search query
    structured_prompt  = f"""
    Original User Query:
    {question}

    --- Dependency Context ---
    {dependency_context}

    --- Prior History ---
    {history_str}
    """

    

    messages = [
        SystemMessage(content="You are a helpful AI expert answering general questions. Your task is to analyze the given input (which may contain both a query , context from previous tools ,prevoious conversation history) and generate a clear and concise answer"),
        HumanMessage(content=structured_prompt)
    ]

    response = llm.invoke(messages)
    return response.content



# print(gen_qna.invoke({
#     "question": "What is blockchain?, context: This is for a fintech investor report focusing on decentralization.",
# }))
