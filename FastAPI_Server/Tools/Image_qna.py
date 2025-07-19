# import boto3
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.tools import tool
# from langchain_core.messages import HumanMessage, SystemMessage
# import pprint

# load_dotenv()

# textract = boto3.client("textract", region_name="us-east-1")

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY")
# )


# @tool
# def image_qna(uploaded_file, query: str):
#     """Use this tool to answer questions from the uploaded image"""
#     if isinstance(uploaded_file, str):
#         with open(uploaded_file, "rb") as file:
#             file_bytes = file.read()
#     else:
#         file_bytes = uploaded_file.read()

#     response = textract.analyze_document(
#         Document={"Bytes": file_bytes}, FeatureTypes=["FORMS", "TABLES"]
#     )
#     extracted_text = ""
#     for block in response["Blocks"]:
#         if block["BlockType"] == "LINE":
#             extracted_text += block["Text"] + "\n"

#     system_prompt = """You are a tool being used by an agent. You will be given the extracted text from an image. 
#         It could be any legal document related to finance, in the form of a fill-up form or just bank statements.
#         Your job is to take this info given to you in the form of text, process it and give answers to the questions asked about the text you have analyzed."""

#     user_prompt = extracted_text + "\n\nQuestion: " + query

#     messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
#     response = llm.invoke(messages)
#     return response.content


# # result = Image_qna.invoke({
# #     "uploaded_file": "/home/saikrishnanair/balancesheet.png",
# #     "query": "What document is this?"
# # })































import boto3
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage,BaseMessage
import pprint
from typing import List, Union


load_dotenv()

textract = boto3.client("textract", region_name="us-east-1")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY")
)


@tool
def image_qna(uploaded_file, query: str,dependency_context: str = "",
    message_history: List[Union[AIMessage, HumanMessage]] = [],):
    """
    Use this tool to answer questions from the uploaded image.
    The query may contain context from previous agent interactions concatenated with the main question and the previous conversation history .
    """

    if isinstance(uploaded_file, str):
        with open(uploaded_file, "rb") as file:
            file_bytes = file.read()
    else:
        file_bytes = uploaded_file.read()

    response = textract.analyze_document(
        Document={"Bytes": file_bytes}, FeatureTypes=["FORMS", "TABLES"]
    )
    extracted_text = ""
    for block in response["Blocks"]:
        if block["BlockType"] == "LINE":
            extracted_text += block["Text"] + "\n"


    def format_history(history: List[BaseMessage]) -> str:
        """Format message history into a string for model input."""
        formatted = ""
        for msg in history[-10:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted += f"{role}: {msg.content}\n"
        return formatted.strip()

    history_str = format_history(message_history)
    question  = f"""
    Original User Query:
    {query}

    --- Dependency Context ---
    {dependency_context}

    --- Prior History ---
    {history_str}
    """


        # Step 1: Use LLM to intelligently parse the query and context
    query_parsing_prompt = SystemMessage(content="""
    You are an expert financial document analyst. Your task is to analyze the given input (which may contain both a query , context from previous tools ,prevoious conversation history) ) and extract the core question about the document.

    The input may be in formats like:
    - "What document is this?"
    - "revenue analysis based on: previous financial analysis showed focus on Q3 metrics"
    - "balance sheet items context: looking for specific ratios from previous step"

    Guidelines:
    1. Parse the input to identify the main question and any contextual information
    2. Extract the core question that needs to be answered about the document
    3. Identify any specific financial metrics, ratios, or analysis focus areas mentioned
    4. Keep context in mind but focus on what specific information is being requested
    5. Be concise and direct

    Output only the refined question - no explanations or additional text.
    """)

    query_parsing_messages = [
        query_parsing_prompt,
        HumanMessage(content=question)
    ]

    refined_query = llm.invoke(query_parsing_messages).content.strip()
    print(f"Refined query: {refined_query}")

    # Step 2: Analyze the document with context-aware system prompt
    system_prompt = """You are a financial document analysis tool being used by an agent system. You will be given extracted text from a financial document image and a question to answer.

    The document could be:
    - Financial statements (balance sheet, income statement, cash flow)
    - Bank statements
    - Investment reports
    - Legal financial documents
    - Fill-up forms with financial data

    Guidelines:
    1. Analyze the extracted text thoroughly
    2. Answer the specific question asked
    3. Keep responses under 150 words and focused
    4. Include specific numbers, dates, and financial metrics when relevant
    5. If the question includes context from previous analysis, incorporate that understanding
    6. Be precise and professional in your analysis

    Focus on providing actionable financial insights."""

    user_prompt = f"Original Input: {query}\nRefined Question: {refined_query}\n\nExtracted Document Text:\n{extracted_text}"

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)

    return response.content

# # Example usage with concatenated context
# result = image_qna.invoke({
#     "uploaded_file": "/home/saikrishnanair/balancesheet.png",
#     "query": "What document is this?"
# })

# print(result)

# Basic usage
# result = image_qna.invoke({
#     "uploaded_file": "/home/saikrishnanair/balancesheet.png",
#     "query": "What document is this?"
# })