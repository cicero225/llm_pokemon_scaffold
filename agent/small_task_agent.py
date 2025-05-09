
# A mini-model (so to speak) for the LLM to call to handle small tasks (like clicking A repeatedly through a dialogue.)
import json

from agent.emulator import Emulator
from agent.utils import convert_tool_defs_to_google_format, convert_tool_defs_to_openai_format, convert_anthropic_message_history_to_google_format, extract_tool_calls_from_gemini
from config import MINI_MODEL, ANTHROPIC_MINI_MODEL_NAME, GEMINI_MINI_MODEL_NAME, OPENAI_MINI_MODEL_NAME, MAX_TOKENS_MINI, TEMPERATURE, MAX_TOKENS_OPENAI
from agent.tool_definitions import PRESS_BUTTON_SCHEMA, TALK_TO_NPC_SCHEMA, NAVIGATE_TO_COORDINATE_SCHEMA

from google.genai import types
from google.genai.errors import ServerError

from openai import BadRequestError
from openai.types import responses

from typing import Any, Optional

# Maybe it's own file if it gets too big, but for now I like just having this here.
SMALL_TASK_TOOL_DEFINITIONS = [
    PRESS_BUTTON_SCHEMA,
    TALK_TO_NPC_SCHEMA,
    NAVIGATE_TO_COORDINATE_SCHEMA,
    {
        "name": "task_done",
        "description": "Call this when you have finished your task",
        "input_schema": {
            "type": "object",
            "properties": {
                "information": {
                    "type": "string",
                    "description": "Any information you want to return (particularly if you were instructed to give it)."
                }
            },
            "required": ["information"],
        },
    },
    {
        "name": "task_aborted",
        "description": "Call this when you have cannot finish the task or have run into serious difficulty",
        "input_schema": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "An explanation of what happened to cause a task abort"
                }
            },
            "required": ["explanation"],
        },
    }
]

MAX_SUBAGENT_STEPS = 50

class SmallTaskAgent:
    def __init__(self, instructions: str, senior_agent: "SimpleAgent", include_text_map: bool, task_id: str, message_history_to_append: str, openai_message_history_to_append: str):
        self.instructions = instructions
        self.history: list[dict[str, Any]] =  []
        # Only now do we clarify the type, to avoid circular imports.
        from agent.simple_agent import SimpleAgent
        self.senior_agent: SimpleAgent = senior_agent  # Just a convenience reference to the calling agent.
        self.tool_defs = SMALL_TASK_TOOL_DEFINITIONS
        if MINI_MODEL == "GEMINI":
            self.tool_defs = convert_tool_defs_to_google_format(SMALL_TASK_TOOL_DEFINITIONS)
        elif MINI_MODEL == "OPENAI":
            self.tool_defs = convert_tool_defs_to_openai_format(SMALL_TASK_TOOL_DEFINITIONS)
        self.needs_text_map = include_text_map
        # deliberately only import this now.
        from agent.simple_agent import logger
        self.logger = logger
        self.task_id = str(task_id)  # The id of the tool call for the senior agent. Used to return eventually.
        self.deferred_information: Optional[dict[str, Any]] = None
        # We're starting to run into pickle limitations. It'd be natural to store a reference to the message history itself, but then it won't pickle the reference unless we also pickle SImpleAgent.
        # and just letting that happen will cause all kinds of hard to diagnose bugs. Instead, we just store the name of the attribute...
        self.message_history_to_append = message_history_to_append
        self.openai_message_history_to_append = openai_message_history_to_append

    def provide_initial_context(self, initial_context: dict[str, Any]):
        self.history.append(initial_context)

    def step(self) -> tuple[bool, Optional[tuple[bool, str]]]:
        """returns should continue subtool, tool finish status, explanation"""
        self.senior_agent.strip_text_map_and_images_from_history(self.history, MINI_MODEL == "OPENAI")
        if len(self.history) > MAX_SUBAGENT_STEPS - 1:
            self.history.append({
                "role": "user",
                "content": "This is a direct order from the developer: You are being timed out. Please report whatever you can and call task_abort."
            })
        malformed = False
        if MINI_MODEL == "CLAUDE":
            response = self.senior_agent.anthropic_client.messages.create(
                model=ANTHROPIC_MINI_MODEL_NAME,
                max_tokens=MAX_TOKENS_MINI,
                system=self.instructions,
                messages=self.history,
                tools=self.tool_defs,
                temperature=TEMPERATURE,
            )
            token_usage = response.usage.input_tokens + response.usage.output_tokens
            self.logger.info(f"Response usage: {token_usage}")
            # Extract tool calls
            tool_calls = [
                block for block in response.content if block.type == "tool_use"
            ]

            # Display the model's reasoning
            for block in response.content:
                if block.type == "text":
                    self.senior_agent.text_display.add_message(f"[Text] {block.text}")
                elif block.type == "tool_use":
                    self.senior_agent.text_display.add_message(f"[Tool] Using tool: {block.name}")

        
            # Process tool calls
            if tool_calls:
                # Add assistant message to history
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({"type": "tool_use", **dict(block)})
        elif MINI_MODEL == "GEMINI":
            # messages -> Gemini format
            google_messages = convert_anthropic_message_history_to_google_format(self.history)
            
            config=types.GenerateContentConfig(
                    max_output_tokens=None,
                    temperature=TEMPERATURE,
                    system_instruction=self.instructions,
                    tools=self.tool_defs  # type: ignore
                )
            chat = self.senior_agent.gemini_client.chats.create(
                model=GEMINI_MINI_MODEL_NAME,
                history=google_messages[:-1],
                config=config
            )   # context caching not available on gemini 2.5. Is it even a good idea for the small task thing?
            retry_limit = 4
            cur_retries = 0
            while cur_retries < retry_limit:
                try:
                    response = chat.send_message(google_messages[-1].parts, config=config)
                    break
                except ServerError as e:
                    if e.code != 500:
                        raise e
                    cur_retries += 1
                except Exception as e:
                    breakpoint()
            self.logger.info(f"Response usage: {response.usage_metadata.total_token_count}")
            tool_calls = []
            assistant_content = []
            if response.candidates is not None:
                text, tool_calls, assistant_content, malformed = extract_tool_calls_from_gemini(response)
                self.senior_agent.text_display.add_message(f"[Text] {text}")
                token_usage = 0  # I didn't even remember to track this but it probably doesn't matter
        elif MINI_MODEL == "OPENAI":
            # OPENAI Specific: Add post_tool call stuff here instead.
            if isinstance(self.history[-1], dict) and self.history[-1].get('type') == "function_call_output":
                self.history.append({
                    "role": "user",
                    "content": self.deferred_information,  # type: ignore
                })

            retries = 2
            cur_tries = 0
            while cur_tries < retries:
                try:
                    response = self.senior_agent.openai_client.responses.create(
                        model=OPENAI_MINI_MODEL_NAME,
                        input=messages_to_use,  # type: ignore
                        instructions=self.instructions,
                        max_output_tokens=MAX_TOKENS_OPENAI,
                        temperature=TEMPERATURE,
                        tools=self.tool_defs
                    )
                    break
                except BadRequestError as e:
                    cur_tries += 1  # Sometimes it spuriously flags this as content violation. I don't know why.
                    breakpoint()
                    continue
                except Exception as e:
                    print(e)
                    breakpoint()
            self.history.extend(response.output)  # type: ignore
            # Gather Reasoning and tool calls
            assistant_content = []
            tool_calls = []
            reasoning_texts = ""
            response_texts = ""
            for chunk in response.output:  # type: ignore
                if isinstance(chunk, responses.ResponseReasoningItem):
                    if chunk.summary:
                        reasoning_texts += " ".join(x.text for x in chunk.summary) + "\n"
                elif isinstance(chunk, responses.ResponseFunctionToolCall):
                    try:
                        assistant_content.append({"type": "tool_use", "id": chunk.call_id, "input": json.loads(chunk.arguments), "name": chunk.name})
                        tool_calls.append(chunk)
                    except Exception as e:
                        breakpoint()
                elif isinstance(chunk, responses.ResponseOutputMessage):
                    try:
                        response_texts += "\n".join(x.text for x in chunk.content)
                    except AttributeError:
                        # This was probably refused for safety reasons. Wait what?
                        breakpoint()
                else:
                    breakpoint()

                full_text = f"<thinking>{reasoning_texts}</thinking>\n\n" if reasoning_texts else "" + response_texts
                if full_text:
                    self.senior_agent.text_display.add_message(f"[Text] {full_text}")
                    assistant_content.append({"type": "text", "text": full_text})
            self.logger.info(f"Response usage: {response.usage.total_tokens if response.usage is not None else 'Unknown'}")
            token_usage = response.usage.total_tokens if response.usage is not None else 0

        # Process tool calls
        if tool_calls: 
            self.history.append(
                {"role": "assistant", "content": assistant_content}
            )
            
            # Process tool calls and create tool results
            tool_results = []
            for tool_call in tool_calls:
                tool_result = self.process_tool_calls(tool_call)
                if not isinstance(tool_result, dict):
                    return False, tool_result
                tool_results.append(tool_result)
                if MINI_MODEL == "OPENAI":
                    openai_result = {
                        "type": "function_call_output",
                        "call_id": tool_result["tool_use_id"],
                        "output": json.dumps(tool_result["content"])
                    }
                    self.history.append(openai_result)

            if malformed:
                tool_results.append({"type": "text", "text": f"WARNING: MALFORMED TOOL CALL. Call using the function call, not in the text."})

            if MINI_MODEL != "OPENAI":
                # Add tool results to message history
                self.history.append(
                    {"role": "user", "content": tool_results}  # type: ignore
                )
        else:
            # Sometimes it just stalls out mysteriously or says some text.
            self.history.append(
                {"role": "user", "content": [{"text": "Can you please continue playing the game?", "type": "text"}]}  # type: ignore
            )
        
        return True, None
            

    def process_tool_calls(self, tool_call) -> tuple[bool, str] | dict[str, Any]:
        """Process a single tool call.
        
        Returns either a dict or True/False (for subtask finish success/failure, with reason)
        """
        tool_name = tool_call.name
        if MINI_MODEL == "CLAUDE":
            tool_input = tool_call.input
            tool_id = tool_call.id
        elif MINI_MODEL == "GEMINI":
            tool_input = tool_call.args
            tool_id = tool_call.id
        elif MINI_MODEL == "OPENAI":
            tool_input = json.loads(tool_call.arguments)
            tool_id = tool_call.call_id
            self.senior_agent.text_display.add_message(f"[Text] {tool_input['explanation_of_action']}")
        self.logger.info(f"Processing tool call: {tool_name}")

        # A limited subset of what's availble to simple agent.
        if tool_name == "press_buttons":
            buttons = tool_input["buttons"]
            wait = tool_input.get("wait", True)
            # Note that this has side effects on location history, etc. But this is DELIBERATE.
            output = self.senior_agent.press_buttons(buttons, wait, tool_id, include_text_map=self.needs_text_map, is_subtool=True)
            if MINI_MODEL == "OPENAI":
                self.deferred_information = output
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {"type": "text", "text": (
                            f"Pressed buttons: {', '.join(buttons)}"
                        )}
                    ],
                }
            else:
                return output
        elif tool_name == "task_done":
            information = tool_input["information"]
            return True, information  # We no longer care to maintain the tool result format, as we will be returning to senior agent now. (though the senior agent certainly needs to keep track.)
        elif tool_name == "task_aborted":
            explanation = tool_input["explanation"]
            return False, explanation
        elif tool_name == "navigate_to_coordinate":
            row = tool_input["row"]
            col = tool_input["col"]
            return self.senior_agent.navigate_to_coordinate(col, row, tool_id, True)
        elif tool_name == "talk_to_npc":
            row = tool_input["row"]
            col = tool_input["col"]
            return self.senior_agent.talk_to_npc(col, row, tool_id, True)
        else:
            breakpoint()
            return False, f"Unknown function call: {tool_name}"