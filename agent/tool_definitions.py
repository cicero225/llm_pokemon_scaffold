
import copy

from google.genai import types

from agent.utils import convert_tool_defs_to_google_format, convert_tool_defs_to_openai_format


PRESS_BUTTON_SCHEMA = {
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

EXPLORE_SCHEMA = {
    "name": "explore_direction",
        "description": "Explore in a direction automatically for a bit. Try this when looking for things or if stuck!",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Direction to prioritize exploring."
                },
            },
            "required": ["direction"],
        },
    }

PRESS_BUTTON_RESTRICTED_SCHEMA = {
        "name": "press_buttons",
        "description": "Press a sequence of buttons on the Game Boy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "buttons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"]
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

TALK_TO_NPC_SCHEMA = {
        "name": "talk_to_npc_or_pick_up_item",
        "description": "Begine talking to a NPC or pick up an item at the coordinates. If the tile is empty, will try to find one within one tile of the given coordinates instead. Will abort if more than one found.",
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "description": "The row coordinate of the NPC."
                },
                "col": {
                    "type": "integer",
                    "description": "The column coordinate of NPC."
                }
            },
            "required": ["row", "col"],
        },
    }

LOG_NPC_SCHEMA = {
        "name": "log_npc_name_and_dialogue",
        "description": "Log who a NPC is at a given location (approximate), along with what they say (which is usually fixed).",
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {
                    "type": "integer",
                    "description": "The row coordinate of the NPC."
                },
                "col": {
                    "type": "integer",
                    "description": "The column coordinate of NPC."
                },
                "name": {
                    "type": "string",
                    "description": "Name of the NPC"
                    
                },
                "dialogue": {
                    "type": "string",
                    "description": "The NPC's dialogue"
                    
                }
            },
            "required": ["row", "col", "name", "dialogue"],
        },
    }

AVAILABLE_TOOLS = [
    PRESS_BUTTON_RESTRICTED_SCHEMA,
    TALK_TO_NPC_SCHEMA,
    LOG_NPC_SCHEMA,
    EXPLORE_SCHEMA
]

AVAILABLE_TOOLS.append({
    "name": "use_subagent",
    "description": "Call another LLM to handle a minor task quickly",
    "input_schema": {
        "type": "object",
        "properties": {
            "detailed_instructions": {
                "type": "string",
                "description": "Detailed instructions to give the other LLM about what its task is"
            },
            "additional_detailed_instructions": {
                "type": "string",
                "description": "Further instructions to give the other LLM about what its task is"
            },
            "return_instructions": {
                "type": "string",
                "description": "Instruction on what information the other LLM should tell you when it's done."
            },
            "needs_text_map": {
                "type": "boolean",
                "description": "Whether this agent needs a copy of the text map to do its work."
            }
        },
        "required": ["detailed_instructions", "additional_detailed_instructions", "return_instructions", "needs_text_map"],
    },
})

"""AVAILABLE_TOOLS.append({
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
})"""

NAVIGATE_TO_COORDINATE_SCHEMA = {
    "name": "navigate_to_coordinate",
    "description": "Will try to take you to a specific coordinate in your text map or on the screenshot. If the exact coordinate is impassable, it will instead try to aim one tile to the side.",
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
}

AVAILABLE_TOOLS.append(NAVIGATE_TO_COORDINATE_SCHEMA)

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
        "name": "detailed_navigator",
        "description": "Ask for help when in a maze-like area. Use if stuck in a loop.",
        "input_schema": {
            "type": "object",
            "properties": {
            },
            "required": [],
        },
})

"""AVAILABLE_TOOLS.append({
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
})"""

GOOGLE_TOOLS = convert_tool_defs_to_google_format(AVAILABLE_TOOLS)

# WHY ARE THEY ALL DIFFERENT AHHHH
OPENAI_TOOLS = convert_tool_defs_to_openai_format(AVAILABLE_TOOLS)

# Just a copy of everything without the detailed_navigator or mark_checkpoint, so it can't call itself or wipe the exploration log.
NAVIGATOR_TOOLS = []
for entry in AVAILABLE_TOOLS:
    if entry["name"] in ["detailed_navigator", "mark_checkpoint", "use_subagent"]:
        continue
    NAVIGATOR_TOOLS.append(entry)

GOOGLE_NAVIGATOR_TOOLS = convert_tool_defs_to_google_format(NAVIGATOR_TOOLS)

OPENAI_NAVIGATOR_TOOLS = convert_tool_defs_to_openai_format(NAVIGATOR_TOOLS)



DISTANT_NAVIGATOR_BUTTONS = [
    {
        "name": "press_buttons",
        "description": "Input a sequence of direction presses",
        "input_schema": {
            "type": "object",
            "properties": {
                "buttons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"]
                    },
                    "description": "List of buttons to press in sequence. Valid buttons: 'up', 'down', 'left', 'right'"
                },
                "wait": {
                    "type": "boolean",
                    "description": "No need to set this. It is true."
                }
            },
            "required": ["buttons"],
        },
    }
]


GOOGLE_DISTANT_NAVIGATOR_BUTTONS = convert_tool_defs_to_google_format(DISTANT_NAVIGATOR_BUTTONS)
OPENAI_DISTANT_NAVIGATOR_BUTTONS = convert_tool_defs_to_openai_format(DISTANT_NAVIGATOR_BUTTONS)