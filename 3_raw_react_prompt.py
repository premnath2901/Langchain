from dotenv import load_dotenv
import re
import inspect
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

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

#ChANGE 3 : Delete the json schemas .Tools now live the  prompt  as plain text 
# We derive description from the function themselves using inspect 

def get_tool_description(tool_dict):
    descriptions = []
    for tool_name, tool_function in tool_dict.items():
        #__wrapped__ bypass decorator  wrapper (e.g @traceable adds *config=Name)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        descriptions.append(f"{tool_name}: {signature}- {docstring}")
    return "\n".join(descriptions)  


tools_description = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES: You must follow theese exactly 
1. NEVER guess or assume any product price you must call product_price tool to get the price of a product
2. Only call apply_discount AFTER you have retrieved the price from the get_product_price tool and pass the exact price  returned by get_product_price -do not pass a made up number 
3. Never Calculate discounts yourself using maths always use apply_discount tool with the exact price returned by get_product_price
4. If user does not specify a discount tier, ask them for which discount tier they want to apply- do not assume one
5. The catalog uses simple product names: laptop, phone, tablet. If the user names one of these and a discount tier, call the tools immediately without asking for confirmation.

Answer the following questions as best you can. You have access to the following tools:

{tools_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:

"""

#CHANGE 4 : Drop tools = from ollama.chat(). The LLM has no idea its an agent --
# all agency comes from prompt above and our regex parsing below 

@traceable(name="Ollama Chat",run_type = "llm")
def ollama_chat_traced(model ,messages , options ):
    return ollama.chat(model=model,messages=messages,options=options)




# --- agent loop with tool calling ---
@traceable(name="Ollama agent loop")
def run_agent(question: str) -> str:
    print(f"    >>> Starting agent loop with question: {question}")
    print("="*60)
    
    #CHANGE 5:One prompt string replaces the system/user message split
    prompt = react_prompt.format(question=question)
    scratchpad = "" 

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n-- -- uteration {iteration} -- --")
        full_prompt = f"{prompt}\n\n{scratchpad}"

        #difference 5:ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation:", "temperature:0"]}
        )
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"--Final Answer--: {ai_message.content}")
            return ai_message.content

        # Process only one tool call per iteration (force)
        # Ollama tool calls have no id: name/arguments live under tool_call.function
        tool_call = tool_calls[0]
        # Difference 6:Attributes access (.function.name)instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"    >>> Tool Call: {tool_name} with args: {tool_args}")

        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in tool_dict")

        # Plain functions (no @tool), so call them directly instead of .invoke()
        #Difference 7: Direct fuction calling instead of .invoke()
        observation = tool_to_use(**tool_args)

        print(f"[Tool Result] {observation}")

        # Keep only the tool call we executed so every call in history has a matching tool message
        messages.append(ai_message.model_copy(update={"tool_calls": [tool_call]}))
        #Difference 8: Tool message format changed to {"role": "tool", "content": str(observation), "tool_name": tool_name}
        messages.append(
            {
                "role": "tool",
                "content": str(observation), 
                "tool_name": tool_name})

    print("ERROR: Max iterations reached without a final answer")
    return None

 



if __name__ == "__main__":

    print("Hello langchain! Let's build an agent that can call tools!(.blind_tool)")
    print("--------------------------------")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
