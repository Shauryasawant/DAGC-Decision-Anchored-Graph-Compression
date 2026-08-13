"""
Real LangChain integration: compress a message list before it reaches the
chat model, using standard LCEL piping.

Requires: pip install "dagc[langchain]"
For this specific example (calling a real model): pip install langchain-openai
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from dagc.integrations.langchain import DAGCCompressor, wrap_chat_model

messages = [
    SystemMessage(content="You are a healthcare assistant. Always verify patient identity."),
    HumanMessage(content="I need to reschedule my physical therapy appointment."),
    AIMessage(content="Please provide your email."),
    HumanMessage(content="dana.brooks@example.com"),
    AIMessage(content="Thanks. Looking up your patient ID."),
    ToolMessage(content="PT-7731Q", tool_call_id="lookup_1"),
    AIMessage(content="Patient verified. Your appointment is Tuesday at 2 PM."),
]

# Option 1: use the compressor standalone.
compressor = DAGCCompressor(target_reduction=0.5)
compressed = compressor.invoke(messages)
print("Standalone:", len(messages), "->", len(compressed), "messages")

# Option 2: pipe it directly in front of a chat model (LCEL).
#
#   from langchain_openai import ChatOpenAI
#   chain = DAGCCompressor(target_reduction=0.85) | ChatOpenAI(model="gpt-4.1")
#   chain.invoke(messages)

# Option 3: wrap an existing model so every .invoke() call is compressed.
#
#   compressed_model = wrap_chat_model(ChatOpenAI(model="gpt-4.1"), target_reduction=0.85)
#   compressed_model.invoke(messages)
