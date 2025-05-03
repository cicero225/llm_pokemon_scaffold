import copy

from typing import Any

from google.genai import types


# Not universal. Just works for what we have in simple_agent.
def convert_anthropic_message_history_to_google_format(messages: list[dict[str, Any]]) -> list[types.Content]:
    google_messages = []
    last_tool_used = None
    last_image_part = None
    for i, message in enumerate(messages):
        role = message["role"]
        if role =="assistant":
            role = "model"
        parts = []
        for idx, content in enumerate(message["content"]):
            if isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            else:
                if content['type'] == 'tool_result':
                    # gemini gets confused sometimes with too many text parts
                    all_text = []
                    for entry in content['content']:
                        if entry['type'] == "text":
                            all_text.append(entry["text"])
                        elif entry['type'] == "image":
                            parts.append(types.Part.from_bytes(data=entry['source']['data'], mime_type=entry['source']['media_type']))
                            last_image_part = parts[-1]
                    all_text = "\n".join(all_text)
                    response_dict = copy.copy(content['content'][0])  # TODO: What if screenshot is first? That never happens but...
                    response_dict['text'] = all_text
                    # We want the text parts in front for google
                    parts = [types.Part.from_function_response(name=last_tool_used, response=response_dict)] + parts
                        
                elif content['type'] == "image":
                    parts.append(types.Part.from_bytes(data=content['source']['data'], mime_type=content['source']['media_type']))
                    last_image_part = parts[-1]
                elif content['type'] == "text":
                    parts.append(types.Part.from_text(text=content['text']))
                    # gemini hack for weird behavior if no image
                    if last_image_part is not None and idx == len(message["content"]) - 1:
                        parts.append(last_image_part)

                elif content['type'] == 'tool_use':
                    last_tool_used = content['name']
                    parts.append(types.Part.from_function_call(name=last_tool_used, args=content['input']))
        google_messages.append(types.Content(parts=parts, role=role))
    return google_messages


# Gemini is derpy (at least how I call it), so getting the tool calls every time is a bit more involved...
def extract_tool_calls_from_gemini(response: types.GenerateContentResponse) -> tuple[str, list[types.FunctionCall], list[dict[str, Any]], bool]:  # output: text, tool_calls (in google format), assistant_content, malformed
    malformed = False
    assistant_content: list[dict[str, Any]] = []
    tool_calls: list[types.FunctionCall] = []
    text = ""
    try:
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                # Gemini is inconsistent about tool use and will sometimes slip into pure text mode
                # inexplicably. If this happens, accommodate it but also berate it for not replyig
                # properly.
                if part.text.startswith("```python"):
                    malformed = True
                    try:
                        _, this_call, _ = part.text.splitlines()
                        try:
                            _, rest = this_call.split(".")
                        except Exception as e:
                            rest = this_call
                        tool_name, this_args = rest.split("(")
                        this_args = this_args.strip(")")
                        split_args = this_args.split("=")
                        # This will be a, "=x, b", "=y, c". We need to resplit.
                        resplit_args = []
                        for idx, x in enumerate(split_args):
                            if idx == 0 or idx == len(split_args) - 1:
                                resplit_args.append(x)
                                continue
                            # split backwards, only taking the first one
                            prev, following = x.rsplit(",", maxsplit=1)
                            resplit_args.append(prev)
                            resplit_args.append(following)
                        dict_args = {}
                        for idx in range(0, len(resplit_args), 2):
                            arg_key = resplit_args[idx].strip()
                            arg_value = resplit_args[idx+1].strip()
                            try:
                                arg_value = int(arg_value)
                            except ValueError as e:
                                # Could be list, in which case probably list of str
                                if arg_value.startswith("["):
                                    arg_value = arg_value[1:-1] # Also dropping "]"
                                    list_arg_value = []
                                    for x in arg_value.split(","):
                                        x = x.strip()
                                        if x.startswith("'") or x.startswith('"'):
                                            x = x[1:-1]
                                        list_arg_value.append(x)
                                    arg_value = list_arg_value
                            dict_args[arg_key] = arg_value
                        assistant_content.append({"type": "tool_use", "id": None, "input": dict_args, "name": tool_name})
                        tool_calls.append(types.FunctionCall(args=dict_args, name=tool_name))
                    except Exception as e:
                        breakpoint()
                else:
                    text += "\n" + part.text
                    assistant_content.append({"type": "text", "text": part.text})
            if part.function_call is not None:
                assistant_content.append({"type": "tool_use", "id": part.function_call.id, "input": part.function_call.args, "name": part.function_call.name})
                tool_calls.append(part.function_call)
    except Exception as e:
        print(e)
        breakpoint()

    return text, tool_calls, assistant_content, malformed


def convert_tool_defs_to_google_format(tool_defs: list[dict[str, Any]]) -> list[types.Tool]:
    GOOGLE_TOOLS: list[types.Tool] = []
    for tool_desc in tool_defs:
        x = copy.copy(tool_desc)  # Shallow copy should be fine here
        x["parameters"] = x["input_schema"]
        del x["input_schema"]
        GOOGLE_TOOLS.append(types.Tool(function_declarations=[x]))
    return GOOGLE_TOOLS

def convert_tool_defs_to_openai_format(tool_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    OPENAI_TOOLS = []
    for tool_desc in tool_defs:
        x = copy.deepcopy(tool_desc)
        x["parameters"] = x["input_schema"]
        del x["input_schema"]
        x["parameters"]["properties"]["explanation_of_action"] = {
                        "type": "string",
                        "description": "MANDATORY: A detailed explanation of why you called this tool"
                    }
        x["parameters"]["required"].append("explanation_of_action")
        x["type"] = "function"
        OPENAI_TOOLS.append(x)
    return OPENAI_TOOLS