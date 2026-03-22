# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from langchain_core.prompts import ChatPromptTemplate

def user_input_formatter():
    system_prompt = """
You are an intent extraction expert.
Your task is only to identify the intent of user message, and to identify the social media platforms where user wants to post.

Platforms to identify
Linkedin
Twitter
Instagram

Rules:
-If a platform is explicitly mentioned, set its value to true.
-If a platform is not mentioned, set its value to false.
- If multiple platform is mentioned, set each of them to true.
- Only detect the intent related to posting on these platforms.
- Do no generate explaination.

Return ONLY valid JSON:
{{
    "linkedin":boolean,
    "twitter":boolean,
    "instagram":boolean,
    "user_message":"user message with platform names removed"
}}

"If the user message does not mention any platform return all values to false"
"""
    return  ChatPromptTemplate([
        ('system',system_prompt),
        ('user',"{input_query}")
    ])

# %%
from langchain.chat_models import init_chat_model
def input_intent_analyst():
    return init_chat_model("groq:llama-3.1-8b-instant", temperature=0)

# %%
chain =  user_input_formatter() | input_intent_analyst() 
chain.invoke({
    "input_query": "I want to post about India 2026 world cup win in linkedin and twitter"
})

# %%
user_input = input("Enter the request:")
chain =  user_input_formatter() | input_intent_analyst() 
chain.invoke({"input_query":user_input})

# %%
def input_node(input_query:str):
    chain = user_input_formatter() | input_intent_analyst()
    return chain.invoke({"input_query":input_query})

# %% [markdown]
# # Langraph Module - Parallel Orchestration

# %%
from typing import List, TypedDict, Annotated
from operator import add, or_
from langgraph.graph import START, END, StateGraph
class InputSchema(TypedDict):
    platform_checker: Annotated[List, add]
    input_query: str
    messages: Annotated[List, add]
    tool_messages: Annotated[List, add]
    search_counts: Annotated[dict, or_]

class OutputSchema(TypedDict):
    final_message: Annotated[List, add]


# %%
import re
import json
def platform_detector_node(state:InputSchema):
   input_state = state
   input_query = state["input_query"]
   response = input_node(input_query)
   response_content = response.content.strip()
   response_content = re.sub(r"^```json\s*|\s*```$", "", response_content, flags=re.MULTILINE)
   data = json.loads(response_content)
   platforms = [{
      "linkedin": data["linkedin"],
      "instagram": data["instagram"],
      "twitter": data["twitter"]
   }]
   input_state["platform_checker"] = platforms
   input_state["messages"] = [{"platform_detector_response":data["user_message"]}]
   return input_state

# %%
def platform_router(state:InputSchema):
    platforms = state["platform_checker"][0]
    return [f"{p}_executor_node" for p,status in platforms.items() if status]

# %%
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults,WikipediaQueryRun
from langchain.tools import tool

@tool
def web_search(query:str):
    """Use this to search latest news on topic requested by user"""
    wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")
    return search.invoke(query)

@tool
def search_wiki(query:str):
    """Used for Wiki Search, """
    wrapper = WikipediaAPIWrapper()
    wikipedia_query = WikipediaQueryRun(api_wapper=wrapper)
    return wikipedia_query.invoke(query)


# %%
from langchain.agents import create_agent
from langgraph.types import Command,Send
from typing import Literal
def linkedin_executor_node(state:InputSchema)->Command[Literal["reducer_node"]]:
    input_state = state
    messages = state["messages"]
    human_msg = next((m for m in messages if 'platform_detector_response' in m),None)
    system_msg = """
        You are a professional LinkedIn content writer. 
        Create a detailed and insightful LinkedIn post about the topic requested by the user. 
        Use a professional tone, include context based on current and historical facts, and structure the post in clear paragraphs suitable for LinkedIn audiences.
    """
    linkedin_agent = create_agent(
        model="groq:llama-3.1-8b-instant",
        tools = [search_wiki, web_search],
        system_prompt=system_msg,
        
    )
    response = linkedin_agent.invoke({
        "messages": [
            {"role": "user", "content": f"Write a 100-line post about {human_msg}"}
        ]
    })
    final_message_obj = response["messages"][-1]
    final_content = final_message_obj.content
    # print("\n--- FINAL LINKEDIN POST ---")
    # print(final_content)
    # print("------------------------------\n")
    if(final_content):
        return Command( update={"final_message": [{"linkedin_final_response": final_content}]},
             goto=["reducer_node"])

        




# %%
def post_prompt_handler():
    return ChatPromptTemplate({
        ('system',"{scope}"),
        ('user',"{human_message}")
    })

def post_generator_builder(scope:str,human_message:str,_tools=None):
    if _tools is None:
        _tools = [web_search]
    llm = input_intent_analyst()
    llm_tools = llm.bind_tools(_tools)
    prompt = post_prompt_handler()
    chain = prompt | llm_tools
    return chain.invoke({
        "scope":scope,
        "human_message":human_message
    })

# %%
def twitter_executor_node(state:InputSchema)->Command[Literal["reducer_node"]]:
    input_state = state
    messages = state["messages"]
    # human_msg = next((m for m in messages if 'platform_detector_response' in m),None)
    system_msg = """
    You are a professional humor writer.
    
    PHASE 1: RESEARCH
    Check if the user's topic requires current facts. If yes, use the ONLY available tool: 'web_search'.
    DO NOT attempt to use any other tools. 
    
    PHASE 2: WRITE
    Once you have the facts (or if none are needed), write a comical one-liner.
    Tone: Witty and sharp.
    Format: Single string for Twitter.
    """
    _search_counts = state["search_counts"] or {}
    print(f"search_counts in twitter node:: {_search_counts}")
    twitter_count = _search_counts.get("twitter", 0)
    print(f"twitter_count in twitter node:: {twitter_count}")
    if twitter_count >= 2:
        system_msg += """
            
            IMPORTANT: You have reached your search limit (2/2). 
            DO NOT use tools again
            Use the existing ToolMessage results and twitter_ai_response in the conversation history to write your final response now.
        """
    print(f"final system message linkedin: {system_msg}")
    print(f"messages sending to twitter llm: {messages}")
    available_tools = [web_search] if twitter_count < 2 else []
    response = post_generator_builder(system_msg,messages,available_tools)
    print(f"Twitter LLM Response:")
    print(response)
    if(response.content):
        return Command( update={"final_message": [{"twitter_final_response": response}]},
            goto=["reducer_node"])
    else:
        state["messages"] = [{"twitter_ai_response":response}]
        return Send("tool_node",[{"current_platform":"twitter"}, {"messages":state["messages"],"search_counts": state["search_counts"]}])

def instagram_executor_node(state:InputSchema)->Command[Literal["reducer_node"]]:
    input_state = state
    messages = state["messages"]
    # human_msg = next((m for m in messages if 'platform_detector_response' in m),None)
    system_msg = """
        You are a professional writer. 
        Create a post about the topic requested by the user. 
        Use a neutral tone, include context based on current facts, and structure the post in two liner for Instagram audiences.
    """
    response = post_generator_builder(system_msg,messages)
    print(f"response::{response}")
    if(response.content):
        return Command( update={"final_message": [{"instagram_final_response": response}]},
            goto=["reducer_node"])
    else:
        state["messages"] = [{"instagram_ai_response":response}]
        return Send("tool_node",[{"current_platform":"instagram"}, {"messages":state["messages"],"search_counts": state["search_counts"]}])

# %%
from langchain_core.messages import ToolMessage
def tool_node(state:InputSchema)->Command[Literal["twitter_executor_node","instagram_executor_node"]]:
    messages = next((d["messages"] for d in state if 'messages' in d),None)
    platform = next((d["current_platform"] for d in state if 'current_platform' in d),None)
    p_name = f"{platform}_ai_response"
    tool_result = []
    _tools = [web_search]
    toolnames = {tool.name: tool for tool in _tools}

    _search_counts = next((d["search_counts"] for d in state if 'search_counts' in d),{platform:0})
    print(f"_search_counts in tool node for {platform}:: {_search_counts}")
    platform_count = _search_counts.get(platform, 0)
    print(f"platform_countin tool node for {platform}:: {platform_count}")
    ai_msg_dict = next((m for m in messages if p_name in m),{})
    ai_msg = ai_msg_dict.get(p_name)
    if ai_msg and hasattr(ai_msg, "tool_calls"):
        for tool_call in ai_msg.tool_calls:
            if platform_count < 2:
                tool = toolnames[tool_call["name"]]
                tool_args = tool_call["args"]
                observation = tool.invoke(tool_args)
                platform_count += 1
            else:
                # Return local limit error to the model
                observation = f"LIMIT REACHED: {platform} has no searches left. Write the post now."
            print(observation)
            tool_result.append(
                (ToolMessage(content=observation, tool_call_id=tool_call["id"]))
            )
    _platform_dict = {platform : platform_count}
    print(f"counts final update in tool node::{_platform_dict[platform]}")
    if platform == 'twitter':
        return Command(
            update={"messages":messages+tool_result, "search_counts": {platform: platform_count} }, goto=["twitter_executor_node"]
        )
    if platform == 'instagram':
        return Command(
            update={"messages":messages+tool_result,"search_counts": {platform: platform_count}}, goto=["instagram_executor_node"]
        )
        
    

# %%

def reducer_node(state:InputSchema):
    all_posts = state.get("final_message", [])  
    print(f"--- Finalizing {all_posts} posts ---")
    return {"final_message": ["Successfully aggregated all platform posts."]}

# %%

graph = StateGraph(InputSchema,input_schema=InputSchema, output_schema=OutputSchema)
graph.add_node("platform_detector_node", platform_detector_node)
graph.add_node("linkedin_executor_node", linkedin_executor_node)
graph.add_node("twitter_executor_node", twitter_executor_node)
graph.add_node("instagram_executor_node", instagram_executor_node)
graph.add_node("reducer_node",reducer_node)
graph.add_node("tool_node",tool_node)

graph.add_edge(START,"platform_detector_node")
graph.add_conditional_edges("platform_detector_node",platform_router)
graph.add_edge("platform_detector_node",END)
# graph.add_edge("platform_detector_node","linkedin_executor_node")
# graph.add_edge("linkedin_executor_node","twitter_executor_node")
# graph.add_edge("twitter_executor_node","instagram_executor_node")
# graph.add_edge("instagram_executor_node","reducer_node")
# graph.add_edge("reducer_node",END)

graph_constructor = graph.compile()
graph_constructor.invoke({
   "input_query":"I need to write a content about India  team in cricket World Cup to post in linkedin and twitter "
})


