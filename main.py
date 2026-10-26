from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool    
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pprint import pprint
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field



class Source(BaseModel):
    """schema for a source used by the agent """
    url: str = Field(..., description="The URL of the source")

class AgentResponse(BaseModel):
    """schema for the response of the agent with answer and sources"""
    answer: str = Field(..., description="The answer to the question")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to answer the question")





tools = [TavilySearch(max_results=5)]
agent = create_agent(model="gpt-4o-mini", tools=tools ,response_format=AgentResponse)

result = agent.invoke({
    "messages": [
        SystemMessage(
            content="You are a helpful assistant that can search the web for information."
        ),
        HumanMessage(content="search 3 job posting related lanchain jobs in linbkedin  ?"),
    ]   
})
pprint(result)
print("--------------------------------")
print(result["messages"][-1].content)
def main() :
    print("Hello from react-search-agent!")

if __name__ == "__main__":
    main()