from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool    
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pprint import pprint
from langchain_tavily import TavilySearch


tavily = TavilySearch(max_results=5)


@tool
def search(query: str) -> str:
    """
    Search the web for the query
    Args:
        query: The query to search for 
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    return tavily.invoke({"query":query})


tools = [search]
agent = create_agent(model="gpt-4o-mini", tools=tools)

result = agent.invoke({
    "messages": [
        SystemMessage(
            content="You are a helpful assistant that can search the web for information."
        ),
        HumanMessage(content="search 3 job posting related lanchain jobs in linbkedin and give me the links in chennai, with location ?"),
    ]   
})
pprint(result)
print("--------------------------------")
print(result["messages"][-1].content)
def main() :
    print("Hello from react-search-agent!")

if __name__ == "__main__":
    main()