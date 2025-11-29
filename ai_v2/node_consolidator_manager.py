import json
from datetime import datetime
from langchain_core.messages import AIMessage
from ai_v2.states import AgentState
from langchain_core.prompts import ChatPromptTemplate
from llm_model import base_llm
from langchain_community.callbacks.manager import get_openai_callback
from langgraph.graph.state import RunnableConfig

def node_consolidator_manager(state: AgentState, config: RunnableConfig) -> AgentState:
    """This node is a consolidator of all raw message coming from the AI and the tools results."""
    
    CONSOLIDATOR_PROMPT = """
Your task is to consolidate and respond naturally based on the user's question, relevant factual 
information from raw_messages, and the conversation summary.

You must ONLY use information that exists in raw_messages. 
Do NOT invent events, people, places, dates, or details that are not explicitly provided.

CONSOLIDATION PRIORITY:
1. Action requests take priority — if raw_messages include requests for information (such as 
   registration details, confirmations, or pending questions), address those first.
2. Maintain conversational flow by integrating pending requests with factual information.
3. Avoid redundancy — do not repeat information unnecessarily.
4. Your answer is what the user will see; respond naturally as if speaking directly to them.
5. Do NOT output reasoning steps, chain-of-thought, or system notes.
6. DO NOT include any information that is not present in raw_messages.

INSTRUCTIONS:
1. Scan raw_messages for any pending actions requiring user attention.
2. Use raw_messages as the ONLY source of factual information. 
   - If it’s not in raw_messages, you cannot mention it.
   - Do NOT create fictional events, locations, names, or details.
3. Use the conversation summary only for context — not for adding new facts.
4. If raw_messages contain details about specific events, consolidate those accurately.
5. If the user asks about events, but raw_messages contain none:
   - You may gently encourage them to check upcoming events.
   - BUT DO NOT describe or invent any event unless raw_messages already contain it.
6. Always respond conversationally and naturally.
7. Use exact names, event titles, dates, venues, and organizer details from raw_messages.
8. If raw_messages do not include a detail the user asked for, say so naturally instead of inventing it.
9. Prioritize addressing any outstanding user questions found inside raw_messages.

⚠️ CRITICAL RULE — STRICT DATE VALIDATION:
- Before inviting the user to any event or suggesting they attend, compare the event date to the current date.
- If the event date is BEFORE the current date:
    - Do NOT invite the user.
    - Instead, state naturally that the event has already taken place.
    - If details exist in raw_messages, summarize them factually.
- If the event date is AFTER or EQUAL to the current date:
    - The event is upcoming; you may invite or encourage participation.
- DO NOT create summaries or descriptions of past events unless the details are explicitly available in raw_messages.

⚠️ ZERO TOLERANCE FOR INVENTION:
- If raw_messages contain 2 events, you only know 2 events.
- If raw_messages contain no confirmed dates, you cannot infer dates.
- If the user asks for something that is missing, respond honestly but politely.

---

Current Date: {current_date_time}

User question: {user_message}

Available information (use only what is here): {raw_messages}

Summary of conversation: {conversation_summary}

Now provide a natural, helpful response following all rules strictly.
"""



#     print(f"[DEBUG] - node_consolidator_manager state: ")
#     print(json.dumps(state, indent=4, ensure_ascii=False, default=str))
    try:        
        raw_messages = ", ".join([item for item in state.get("raw_messages", [])])

        # print(f"[DEBUG] - node_consolidator_manager raw_messages: {raw_messages}")
        # Create persuasion tools 
        user_message = state.get("input_message", "")
        # Create Elena agent
        persuasion_prompt = ChatPromptTemplate.from_template(CONSOLIDATOR_PROMPT)                  
        
        chain = persuasion_prompt | base_llm
        
        with get_openai_callback() as cb:

            start_time = datetime.now()
            try:
                llm_output = chain.invoke({
                    "raw_messages": raw_messages,
                    "user_message": user_message,
                    "conversation_summary": state.get("short_message",""),
                    "current_date_time": datetime.now().isoformat()
                }, config=config)      
            except Exception as e:
                # LangGraph provides more specific exception types
                print(e)
                llm_output = AIMessage(content="I need a more specific question to help you.")
               
            # Execute persuasion agent
            print("**********************************")
            print("--- node_consolidator_manager ---")
            print(f"Prompt tokens: {cb.prompt_tokens}")
            print(f"Completion tokens: {cb.completion_tokens}")
            print(f"Total tokens: {cb.total_tokens}")
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\nTime spent node_consolidator_manager: {elapsed:.3f}")            
            print("**********************************")
            return {
                **state,
                "messages":llm_output
            }
        
    except Exception as e:
        print(f"Error in node_consolidator_manager: {e}")       
        return {
            **state,
            "messages": AIMessage(content="Sorry, I'm having trouble consolidating the messages right now. Please try again later.")    
        }