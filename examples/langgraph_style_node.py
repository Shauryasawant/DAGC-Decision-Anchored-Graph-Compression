"""
Demo of using DAGC as a LangGraph-style node.
"""

from dagc import compress


def compress_context_node(state: dict) -> dict:
    """Compress messages before sending them to the LLM."""
    state["messages"] = compress(
        state["messages"],
        target_reduction=0.50
    )
    return state


# Example state; LangGraph normally provides this.

state = {
    "messages": [
        {
            "role": "system",
            "content": "You are a healthcare assistant. Always verify patient identity."
        },
        {
            "role": "user",
            "content": "I need to reschedule my physical therapy appointment."
        },
        {
            "role": "assistant",
            "content": "Please provide your email."
        },
        {
            "role": "user",
            "content": "dana.brooks@example.com"
        },
        {
            "role": "assistant",
            "content": "Thanks. Looking up your patient ID."
        },
        {
            "role": "tool",
            "content": "PT-7731Q"
        },
        {
            "role": "assistant",
            "content": "Patient verified. Your appointment is Tuesday at 2 PM."
        }
    ]
}

print("Before compression:", len(state["messages"]), "messages")

state = compress_context_node(state)

print("After compression:", len(state["messages"]), "messages")
print()

for msg in state["messages"]:
    print(f"{msg['role']:10} {msg['content']}")
