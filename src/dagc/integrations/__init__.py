"""
dagc.integrations — optional framework adapters for LangChain and LangGraph.

These modules are import-optional: importing `dagc` never pulls in
LangChain, and importing `dagc.integrations.langchain` without
`langchain-core` installed raises a clear ImportError telling the user
what to install, instead of a confusing traceback.

    pip install "dagc[langchain]"     # LCEL-style message compressor
    pip install "dagc[langgraph]"     # StateGraph compression node
"""
