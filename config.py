from typing import Literal

# Original configuration for the application. Unfortunately I am not very disciplined so only a small amount of settings are here.
ANTHROPIC_MODEL_NAME = "claude-3-7-sonnet-20250219"
GEMINI_MODEL_NAME = "gemini-2.5-pro-preview-03-25"  # Or, for example, "gemini-2.5-flash-preview-04-17"
OPENAI_MODEL_NAME = "o3"  # o4-mini also works.

# This configures what family of models we end up using.
MODEL: Literal["CLAUDE", "GEMINI", "OPENAI"] = "CLAUDE"

# Currently only for the unused navigation_assistance feature. Can be ignored.
MAPPING_MODEL: Literal["CLAUDE", "GEMINI", "OPENAI"] = "CLAUDE"


TEMPERATURE = 1.0
MAX_TOKENS = 10000

# bypass using Claude for "navigate_to_offscreen_coordinate" because it's really token-expensive and also we know it can reliably do it if we give a HUGE amount of tokens,
# so it doesn't prove much. (plus I'd have to do streaming which is just annoying)
# This basically saves money at the cost of being a bit unsatisfying. This is a lot faster though.
DIRECT_NAVIGATION = True