
import copy

from google.genai import types

AVAILABLE_TOOLS = [
    {
        "name": "press_buttons",
        "description": "Press a sequence of buttons on the Game Boy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "buttons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["a", "b", "start", "select", "up", "down", "left", "right"]
                    },
                    "description": "List of buttons to press in sequence. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'"
                },
                "wait": {
                    "type": "boolean",
                    "description": "Whether to wait for a brief period after pressing each button. Defaults to true."
                }
            },
            "required": ["buttons"],
        },
    }
]

AVAILABLE_TOOLS.append({
    "name": "navigate_to",
    "description": "Automatically navigate to a position on screen. The available locations are included in your screenshot. This tool is only available in the overworld.",
    "input_schema": {
        "type": "object",
        "properties": {
            "row": {
                "type": "integer",
                "description": "The row coordinate to navigate."
            },
            "col": {
                "type": "integer",
                "description": "The column coordinate to navigate."
            }
        },
        "required": ["row", "col"],
    },
})

AVAILABLE_TOOLS.append({
            "name": "bookmark_location_or_overwrite_label",
            "description": "Label a location you have been to with a useful label for later consideration. For instance, 'Entrance to Area' or 'Stairs to first floor'. ALSO, use this to overwrite incorrect labels.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location you are labeling"
                    },
                    "col": {
                        "type": "integer",
                        "description": "The column coordinate to label (0-9)."
                    },
                    "row": {
                        "type": "integer",
                        "description": "The row coordinate to label (0-8)."
                    },
                    "label": {
                        "type": "string",
                        "description": "The label to assign this location. NOTE: KEEP THIS SHORT"
                    },
                },
                "required": ["location", "col", "row", "label"],
            },
})

AVAILABLE_TOOLS.append({
        "name": "mark_checkpoint",
        "description": "Called when succeeding a navigational task OR blacking out, to reset the step counter.",
        "input_schema": {
            "type": "object",
            "properties": {
                "achievement": {
                    "type": "string",
                    "description": "What did you successfully achieve?"
                },
            },
            "required": [],
        },
})

AVAILABLE_TOOLS.append({
        "name": "navigation_assistance",
        "description": "Ask for navigation advice on a difficult navigation task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "navigation_goal": {
                    "type": "string",
                    "description": "What the current goal is to ask the navigation assistance tool for help on."
                },
            },
            "required": ["navigation_goal"],
        },
})




GOOGLE_TOOLS = []
for tool_desc in AVAILABLE_TOOLS:
    x = copy.copy(tool_desc)  # Shallow copy should be fine here
    x["parameters"] = x["input_schema"]
    del x["input_schema"]
    GOOGLE_TOOLS.append(types.Tool(function_declarations=[x]))

# WHY ARE THEY ALL DIFFERENT AHHHH
OPENAI_TOOLS = []
for tool_desc in AVAILABLE_TOOLS:
    x = copy.copy(tool_desc)  # Shallow copy should be fine here
    x["parameters"] = x["input_schema"]
    del x["input_schema"]
    x["parameters"]["properties"]["explanation_of_action"] = {
                    "type": "string",
                    "description": "MANDATORY: A detailed explanation of why you called this tool"
                }
    x["parameters"]["required"].append("explanation_of_action")
    x["type"] = "function"
    OPENAI_TOOLS.append(x)