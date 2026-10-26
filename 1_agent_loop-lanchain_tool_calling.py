from dotenv import load_dotenv

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable 
from langchain_ollama import ChatOllama

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product_name: str) -> float:
    """Get the price of a product"""
    print(f"    >>> Executing tool get_product_price for product: {product_name}")
    price = { "laptop": 1000, "phone": 500, "tablet": 300 }
    return price.get(product_name, 0)

    
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price
      available_discount_tiers = ["bronze", "silver", "gold"]"""
    print(f"    >>> Executing tool apply_discount for price: {price} and discount_tier: {discount_tier}")
    discount_tiers = {"bronze": 5, "silver": 12, "gold": 20}
    discount = discount_tiers.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- agent loop with tool calling ---
@traceable(name="Langchain Tool Calling Agent")
def run_agent(question: str) -> str:
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}

    llm = ChatOllama(model="qwen3:1.7b" , temperature=0)
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
                """
            )),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n-- -- uteration {iteration} -- --")
        
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = llm_with_tools.get_tool_calls(ai_message)

        if not tool_calls:
            print(f"--Final Answer--: {ai_message.content}")
            return ai_message.content
  
        # Process only one tool call per iteration (force)
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_call_id = tool_call.get("id")

        print(f"    >>> Tool Call: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in tools_dict")

        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] {observation}")
      

  
    
    for i in range(MAX_ITERATIONS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        print(f"    >>> Iteration {i+1} response: {response.content}")

 



if __name__ == "__main__":

    print("Hello langchain! Let's build an agent that can call tools!(.blind_tool)")
    print("--------------------------------")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
