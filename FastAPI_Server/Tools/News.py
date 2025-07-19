import os
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage,BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from dotenv import load_dotenv
from pprint import pprint
import trafilatura
from typing import List, Union
load_dotenv()

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

@tool
def financial_news_search(query: str,dependency_context: str = "",
    message_history: List[Union[AIMessage, HumanMessage]] = [],) -> str:
    """
    This tool is used to answer queries which needs latest news feed.
    Uses LLM to intelligently formulate search queries from concatenated query+context strings.
    
    Args:
        query (str): The query string that may include context from previous agents
    """
    print("news tool invoked")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))
    def format_history(history: List[BaseMessage]) -> str:
        """Format message history into a string for model input."""
        formatted = ""
        for msg in history[-10:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted += f"{role}: {msg.content}\n"
        return formatted.strip()

    history_str = format_history(message_history)
    try:
        # Step 1: Use LLM to intelligently formulate the search query
        question  = f"""
    Original User Query:
    {query}

    --- Dependency Context ---
    {dependency_context}

    --- Prior History ---
    {history_str}
    """

        
        query_formulation_prompt = SystemMessage(content="""
You are an expert financial research assistant. Your task is to analyze the given input (which may contain both a query , context from previous tools ,prevoious conversation history) and generate an optimized search query for financial news.

The input may be in formats like:
- "what is the current price of solana?"
- "sector analysis based on: [previous document analysis about semiconductor industry]"
- "market trends context: renewable energy consolidation analysis from previous step"

                                                 

Guidelines:
1. Parse the input to identify the core query and any contextual information
2. Extract key financial entities, companies, sectors, or concepts
3. Identify the most relevant financial aspects (earnings, market trends, regulatory changes, etc.)
4. Focus on recent developments and market-moving events
5. Use specific financial terminology that would appear in news articles
6. Synthesize context with the main query to create a targeted search
7. Avoid generic terms and focus on actionable, newsworthy elements
                                                 

FORMULATE THE QUERY BASED ON ONLY THE KEYWORDS LIKE FINANICAL TERMS , COMPANY NAMES ETC..
                                                

Output only the optimized search query - no explanations or additional text.
""")
        
        query_formulation_messages = [
            query_formulation_prompt,
            HumanMessage(content=question)
        ]
        
        optimized_query = llm.invoke(query_formulation_messages).content.strip()
        print(f"Optimized search query: {optimized_query}")
        
        # Step 2: Search using the optimized query
        enhanced_query = f"latest financial news on {optimized_query}"
        response = tavily_client.search(
            query=enhanced_query,
            topic="finance",
            time_range="month",
            max_results=3,
            country="India",
            include_domains=[
                "financialexpress.com",
                "economictimes.indiatimes.com",
                "livemint.com",
                "thehindu.com",
                "wionews.com",
                "moneycontrol.com",
                "business-standard.com",
                "reuters.com",
                "bloomberg.com"
            ],
            exclude_domains=[
                "reddit.com",
                "twitter.com",
                "facebook.com",
                "X.com",
                "instagram.com",
                "youtube.com"
            ]
        )
        
        if not response.get('results'):
            return f"No recent financial news found for: {optimized_query}"
        
        # Step 3: Extract and process content
        urls = [result.get('url') for result in response['results'] if result.get('url')]
        extracted_text = ""
        successful_extractions = []
        
        for url in urls:
            try:
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    print(f"❌ Could not fetch: {url}")
                    continue

                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    include_formatting=False,
                    date_extraction_params={"extensive_search": True}
                )
                
                if text:
                    extracted_text += f"\n\n--- Article from {url} ---\n{text}"
                    successful_extractions.append(url)
                else:
                    print(f"❌ Could not extract content from: {url}")
                    
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                continue
        
        if not extracted_text:
            return f"Could not extract content from any of the found articles for: {optimized_query}"
        
        # Step 4: Use LLM to analyze and present the content
        analysis_prompt = SystemMessage(content="""
You are a financial news analyst. Analyze the provided news articles and give a concise, focused summary.

Guidelines:
1. Keep response under 200 words
2. Focus only on key financial insights and market-moving information
3. Use bullet points for main facts
4. Include specific numbers, dates, and percentages when available
5. Prioritize recent developments and actionable insights
6. Remove fluff and irrelevant details

Be concise and direct - focus on what matters most for financial decision-making.
""")
        
        analysis_input = f"Original Input: {query}\nOptimized Search: {optimized_query}\n\nArticle Content:\n{extracted_text}"
        
        analysis_messages = [
            analysis_prompt,
            HumanMessage(content=analysis_input)
        ]
        
        analysis_result = llm.invoke(analysis_messages).content
        
        # Step 5: Format final result
        final_result = f"""
🔍 **Search Query Used**: {optimized_query}

{analysis_result}

📚 **Sources**:
{chr(10).join(f"• {url}" for url in successful_extractions)}
        """
        
        return final_result.strip()
        
    except Exception as e:
        error_msg = f"Error in financial news search: {str(e)}"
        print(f"{error_msg}")
        return error_msg

# Example usage with concatenated context
# print(financial_news_search.invoke({
#     "query": "what is the current price of solana? Context: Previous analysis showed interest in cryptocurrency market trends and DeFi sector growth"
# }))

# Example of how Router might call this with concatenated context
# query_with_context = f"sector analysis based on: {document_qna_output}"
# result = financial_news_search.invoke({"query": query_with_context})