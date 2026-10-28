from dotenv import load_dotenv

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable 
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import ollama
import os

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(run_type = "tool")
def get_product_price(product_name: str) -> float:
    """Get the price of a product"""
    print(f"    >>> Executing tool get_product_price for product: {product_name}")
    price = { "laptop": 1000, "phone": 500, "tablet": 300 }
    return price.get(product_name, 0)

    
@traceable(run_type = "tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price
      available_discount_tiers = ["bronze", "silver", "gold"]"""
    print(f"    >>> Executing tool apply_discount for price: {price} and discount_tier: {discount_tier}")
    discount_tiers = {"bronze": 5, "silver": 12, "gold": 20}
    discount = discount_tiers.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


#Difference  2: without @tool, we must manually define the JSON schema for each function 
#This is exactly what Langchain's @tool decorator generates automatically 
#from the function's type hints and docstring 

tools_for_llm = [
    {
        "type":"function",
        "function": {
            "name": "get_product_price",
            "description": "Get the price of a product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The name of the product"
                    },
                },
            },
            "required": ["product_name"],
        }
    },
    {
        "type":"function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount to a price",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The price of the product"
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply"
                    },
                },
            },
            "required": ["price", "discount_tier"],
        }
    }
]

#Note : Ollama can also auto-generate these schemas if pass the function 
#directly as tool (similar to LangChain's @tool decorator)
#tools_for_llm=[get_product_price, apply_discount]
#However, this reuires your docstrings to follow  the Google docstring  format 
# So ollama can parse parameter description from the Args section , For example 
# def get_product_price(product_name: str) -> float:
#    """Looks up the price of a product in the catalog"""
#    Args:
#        product_name: The name of the product to look up
#    Returns:
#        The price of the product
#    """
# We keep the manula json versuon here , so you can see what @tool hide from you

# ---- Helper:traced ollama call ----
# Difference 3:without langchain  , we must trace llm calls for Lnagsmith 
@traceable(name="Ollama Chat",run_type = "llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL,messages=messages,tools=tools_for_llm)

    




# --- agent loop with tool calling ---
@traceable(name="Langchain Tool Calling Agent")
def run_agent(question: str) -> str:
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}

    llm = ChatOllama(model="qwen3:1.7b" , temperature=0)
    # llm = ChatOpenAI(
    # model="openai/gpt-4o-mini",
    # api_key=os.getenv("OPENROUTER_API_KEY"),
    # base_url="https://openrouter.ai/api/v1",
    # temperature=0,
    # )

    llm_with_tools = llm.bind_tools(tools)

    print(f"    >>> Starting agent loop with question: {question}")
    print("="*60)
    
    messages = [
        SystemMessage(
            content=(
                """
                 You are the helpfull shopping assistant
                 you can access to a product catalog tool 
                 and a discount  tool
                 STRICT RULES: You must follow theese exactly 
                 1.NEVER guess or assume any product price
                 you must call product_price tool to get the price of a product
                 2.Only call apply_discount AFTER you have retrieved the price from the get_product_price tool
                 and pass the exact price 
                 returned by get_product_price -do not pass a made up number 
                 3. Never Calculate discounts yourself using maths
                 always use apply_discount tool with the exact price returned by get_product_price
                 4. If user does not specify a discount tier, ask them for which discount tier they want to apply- do not assume one
                 5. The catalog uses simple product names: laptop, phone, tablet. If the user names one of
   these and a discount tier, call the tools immediately without asking for confirmation.

                """
            )),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n-- -- uteration {iteration} -- --")
        
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"--Final Answer--: {ai_message.content}")
            return ai_message.content
  
        # Process only one tool call per iteration (force)
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_call_id = tool_call.get("id")

        print(f"    >>> Tool Call: {tool_name} with args: {tool_args}")

        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in tool_dict")

        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation}")

        # Keep only the tool call we executed so every call in history has a matching ToolMessage
        messages.append(ai_message.model_copy(update={"tool_calls": [tool_call]}))
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Max iterations reached without a final answer")
    return None

 



if __name__ == "__main__":

    print("Hello langchain! Let's build an agent that can call tools!(.blind_tool)")
    print("--------------------------------")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
