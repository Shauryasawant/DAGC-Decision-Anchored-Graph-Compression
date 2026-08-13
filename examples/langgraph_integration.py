"""
Real LangGraph integration: a compression node wired into a StateGraph.

Unlike examples/langgraph_style_node.py (which demos the idea on a plain
dict), this uses dagc.integrations.langgraph so the compression node
actually shrinks `MessagesState.messages` -- correctly emitting
RemoveMessage entries so LangGraph's add_messages reducer doesn't just
append the compressed output on top of the originals.

Requires: pip install "dagc[langgraph]"
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from dagc.integrations.langgraph import make_compression_node


def fake_agent_node(state: MessagesState) -> dict:
    """Stand-in for a real LLM call; just echoes an assistant reply."""
    return {"messages": [AIMessage(content="Got it, appointment confirmed for Tuesday 2 PM.")]}


graph = StateGraph(MessagesState)
graph.add_node("compress", make_compression_node(target_reduction=0.6, min_messages_to_compress=3))
graph.add_node("agent", fake_agent_node)
graph.add_edge(START, "compress")
graph.add_edge("compress", "agent")
graph.add_edge("agent", END)

app = graph.compile()

initial_state = {
    "messages": [
        HumanMessage(content="I need to reschedule my physical therapy appointment."),
        AIMessage(content="Sure, can you confirm your email?"),
        HumanMessage(content="dana.brooks@example.com"),
        AIMessage(content="Thanks, looking up your patient ID now."),
        ToolMessage(content="PT-7731Q", tool_call_id="lookup_1"),
    ]
}

if __name__ == "__main__":
    print("Before:", len(initial_state["messages"]), "messages")
    result = app.invoke(initial_state)
    print("After: ", len(result["messages"]), "messages")
    for m in result["messages"]:
        print(f"{m.type:10} {m.content}")
