from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal, cast

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware.types import InputAgentState
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
import re
import operator
from schemas import (
    UserIntent,
    SessionState,
    AnswerResponse,
    SummarizationResponse,
    CalculationResponse,
    UpdateMemoryResponse,
)
from prompts import (
    get_intent_classification_prompt,
    get_chat_prompt_template,
    MEMORY_SUMMARY_PROMPT,
)



from pprint import pprint 


# TODO: The AgentState class is already implemented for you.  Study the
# structure to understand how state flows through the LangGraph
# workflow.  See README.md Task 2.1 for detailed explanations of
# each property.
class AgentState(TypedDict, total=False):
    """
    The agent state object
    """

    # Current conversation
    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str

    # Memory and context
    conversation_summary: str
    active_documents: Optional[List[str]]

    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    # Session management
    session_id: Optional[str]
    user_id: Optional[str]

    actions_taken: Annotated[List[str], operator.add]


def invoke_react_agent(
    response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools
) -> tuple[Dict[str, Any], List[str | None]]:
    llm_with_tools = llm.bind_tools(tools)

    agent = create_agent(
        model=llm_with_tools,  # Use the bound model
        tools=tools,
        response_format=response_schema,
    )

    agent_input = cast(InputAgentState, {"messages": messages})
    result = agent.invoke(agent_input)
    tools_used = [
        t.name for t in result.get("messages", []) if isinstance(t, ToolMessage)
    ]

    tools_used = [eachtool for eachtool in tools_used if eachtool in [t.name for t in tools]]

    return result, tools_used


def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        raise ValueError("configurable not found in config")

    llm = configurable.get("llm")
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be a ChatOpenAI instance")

    llm_with_structured_output = llm.with_structured_output(UserIntent)

    prompt = get_intent_classification_prompt().format(
        user_input=state.get("user_input"),
        conversation_history=state.get("messages", []),
    )

    response = llm_with_structured_output.invoke(prompt)
    if not isinstance(response, UserIntent):
        raise TypeError("LLM response is not of type UserIntent")

    conditional_logic: Dict[str, str] = {
        "qa": "qa_agent",
        "summarization": "summarization_agent",
        "calculation": "calculation_agent",
    }

    intent = response.intent_type
    next_step = conditional_logic.get(intent, "qa_agent")


    return {
        "actions_taken": ["classify_intent"],
        "intent": response,
        "next_step": next_step,
    }


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        raise ValueError("configurable not found in config")

    llm = configurable.get("llm")
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be a ChatOpenAI instance")

    tools = configurable.get("tools")
    if not isinstance(tools, list):
        raise TypeError("tools must be a list")

    prompt_template = get_chat_prompt_template("qa")

    user_input = state.get("user_input")
    if not isinstance(user_input, str):
        raise TypeError("user_input must be a string")

    messages = prompt_template.invoke(
        {
            "input": user_input,
            "chat_history": state.get("messages", []),
        }
    ).to_messages()

    result, tools = invoke_react_agent(AnswerResponse, messages, llm, tools)
    tools_used: list[str] = [tool for tool in tools if tool is not None]

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["qa_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        raise ValueError("configurable not found in config")

    llm = configurable.get("llm")
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be a ChatOpenAI instance")

    tools = configurable.get("tools")

    prompt_template = get_chat_prompt_template("summarization")
    messages = prompt_template.invoke(
        {
            "input": state.get("user_input"),
            "chat_history": state.get("messages", []),
        }
    ).to_messages()

    response, tools_used = invoke_react_agent(
        SummarizationResponse, messages, llm, tools
    )
    structurized_response = response.get("structured_response")
    if not isinstance(structurized_response, SummarizationResponse):
        raise TypeError("LLM response is not of type SummarizationResponse")

    return {
        "messages": response.get("messages",[]),
        "actions_taken": ["summarization_agent"],
        "current_response": response,
        "tools_used": tools_used,  
        "next_step": "update_memory",
    }


# TODO: Implement the calculation_agent function. Refer to README.md Task 2.3
def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        raise ValueError("configurable not found in config")

    llm = configurable.get("llm")
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be a ChatOpenAI instance")

    tools = configurable.get("tools")

    prompt_template = get_chat_prompt_template("calculation")
    messages = prompt_template.invoke(
        {
            "input": state.get("user_input"),
            "chat_history": state.get("messages", []),
        }
    ).to_messages()

    response, used_tools = invoke_react_agent(
        CalculationResponse, messages, llm, tools
    )

    structurized_response = response.get("structured_response")
    pprint(structurized_response)

    if not isinstance(structurized_response, CalculationResponse):
        raise TypeError("LLM response is not of type Calculation")

    return {
        "messages": response.get("messages", []),
        "actions_taken": ["calculation_agent"],
        "current_response": response,
        "tools_used": used_tools,
        "next_step": "update_memory",
    }


def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """

    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        raise ValueError("configurable not found in config")

    llm = configurable.get("llm")
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be a ChatOpenAI instance")

    prompt_with_history = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
            MessagesPlaceholder("chat_history"),
        ]
    ).invoke(
        {
            "chat_history": state.get("messages", []),
        }
    )

    structured_llm = llm.with_structured_output(SummarizationResponse)

    response = structured_llm.invoke(prompt_with_history)

    if not isinstance(response, SummarizationResponse):
        raise TypeError("LLM response is not of type SummarizationResponse")

    return {
        "conversation_summary": response.summary,
        "active_documents": response.document_ids,
        "next_step": "end",
    }


def should_continue(state: AgentState) -> str:
    """Router function"""
    return state.get("next_step", "end")


# TODO: Complete the create_workflow function. Refer to README.md Task 2.5
def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """
    workflow = StateGraph(AgentState)

    # TODO: Add all the nodes to the workflow by calling workflow.add_node(...)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
            "end": END,
        },
    )

    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")

    workflow.add_edge("update_memory", END)

    # TODO Modify the return values below by adding a checkpointer with InMemorySaver
    return workflow.compile(checkpointer=InMemorySaver())
