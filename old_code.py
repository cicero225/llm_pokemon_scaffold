# Maybe you will find useful treasure in here. Who knows?


MAPPING_CORRECTION_PROMPT = """
Your job is to correct a given ASCII map with a mistake in it.

You will be given the ASCII map and a text instruction on how to fix it.

Please output a corrected map. If the map has other text commentary in it, leave it alone and emit it again verbatim.
"""

MAPPING_PROMPT_SCREENSHOT = """Your job is to act as mapping agent for another agent playing through Pokemon Red.

You will be shown a screenshot of Pokemon Red and will be possibly be provided with an existing ASCII map of the area, drawn by you previously, with the previous suggestion.
You may also be provided with a collision map of the current screen based on the game's RAM.

PRIORITY ONE: if the screenshot you are given is NOT of a mappable part of Pokemon Red (for example, you're inside a menu or the start of the game), SIMPLY RETURN THE EXISTING MAP.
PRIORITY TWO: EXTEND the existing map to cover what you see on screen/what is in the collision map.
PRIORITY THREE: If it is apparent from the screenshot that all or most reachable tiles have been explored (only a few cyan tiles are visible), add a direct command to leave the immediate area, as the current area is not correct for progress.

Your job is to (if necessary) EXTEND the ASCII map to account for the new information.

Keep the map SMALL if possible. For instance, an entire game screen may be about twice the size of this example below:

(1,3)
|                 |
|█P(3,4)██████████| P = Path
|                 |
|   S(5,6)        | S = Sign
|                 |
|██P(7,6) ████████|
| █P(8,6) █       |
| █P(9,6) █       |
| █P(10,6)█       |
                  (11, 13) 

It will not be easy to make a good map like this, but do your best.              

The screenshot will contain the following extra information to help guide you:

1. RED squares are IMPASSIBLE, but may sometimes contain doors, stairs, or other warp points, so if you see one you may still walk through. Ledges can sometimes be jumped from above.
Buildings always have red squares on them, because they are impassible.
2. CYAN squares are PASSABLE and represent potential paths.
3. MAROON squares are NPCs or Items on the ground (in a round pokeball). If it's not MAROON it's not a NPC!
4. GREEN squares are the player's last few locations, showing what direction they came from. Try to go to new places!
5. BLUE squares are locations you visited during the current location tracking, labeled by the location tracker.
        Make sure to strongly prioritize locations NOT IN BLUE, as this means you haven't explored there yet.

It will also contain a system of grid coordinates, which is consistent throughout the region.

1. The top-left corner of the location is at 0, 0
2. The first number is the column number, increasing horizontally to the right.
3. The second number is the row number2 increasing vertically downward.

Keep the following in mind:

1. The grid coordinates are an excellent guide for mapping, and you should include many of these onto your map.
    1a. Make sure your map is internally consistent.
2. Label the locations of buildings, BUT DO NOT NAME THE BUILDING UNLESS IT ALREADY HAS A LABEL IN THE SCREENSHOT.
3. Buildings are ALMOST ALWAYS large squarish groupings labeled with MULTIPLE red squares (at least 9). Do not label
   anything with a single red square or a line of red squares as a building--these are instead walls, fences, etc.
3. Also label anything that already has a label on the screen. i.e., you may see text on the screen identifying the
   a building or NPC. Make sure to include this if so.

HIGH PRIORITY: Unless PRIORITY THREE is in action, say NOTHING OTHER THAN THE MAP.
If PRIORITY THREE is active (There are almost no cyan squares on screen) then instead add the following below the map:

EXPLORATION COMMAND: [Instruction to leave the immediate area and head for cyan squares]

NOTE: DO NOT instruct leaving the location entirely. Just somewhere offscreen. Do not suggest directions.
"""

MAPPING_PROMPT_NO_SCREENSHOT = """Your job is to act as mapping agent for another agent playing through Pokemon Red.

You will possibly be provided with an existing ASCII map of the area, drawn by you previously, with the previous suggestion.
You wiil also be provided with a collision map of the current screen based on the game's RAM, along with coordinates for the corners.

EXTEND the existing map to cover what is in the new collision map.

Here is how the grid coordinates work:

1. The top-left corner of the location is at 0, 0
2. The first number is the column number, increasing horizontally to the right.
3. The second number is the row number2 increasing vertically downward.
"""


UNUSED = """
4. MAKE SURE to label the edge of the map (which inside buildings will typically be black with red squares).

Finally, after drawing your map, include a small chunk of text like so:

EXPLORATION SUGGESTION: [suggestion]

Follow the following rules:

1. IF THE ENTIRE BUILDING OR LOCATION HAVE BEEN MAPPED:
   Say: EXPLORATION SUGGESTION: No Suggestion
GOOD EXAMPLE (where the map does not have information about the top left): The area near the top left past the pokemon center has not been mapped. It may be worth taking a look.
BAD EXAMPLE (for a building with a full map):  You have not explored the top-left corner near (1,1) to (3,3), nor the area near the exit mat in row 7.

    In the Bad example above, you are encouraging repetitive exploration of the same areas.

DO NOT SAY ANYTHING OTHER THAN basic exploration comments, as you do not have full context.
GOOD EXAMPLE: You have not explored above the path at (3,4) yet. This may be worth doing.
BAD EXAMPLE: You are near Giovanni, and this is Viridian Gym. You should...    
"""


def mapping_tool(self, include_screenshot: bool=True, use_collision_map=False) -> Optional[str]:
        logger.info(f"[Agent] Running Mapping Tool...")
        
        _, location, coords = self.emulator.get_state_from_memory()

        if include_screenshot:
            # Get a fresh screenshot after executing the buttons
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=2, add_coords=True, player_coords=coords, location=location)

        this_map = self.map_tool_map.get(location)
        if this_map is None:
            mapping_query = f"""There is no preexisting map. Please start making a new one!
            
            Current location: {location}
"""
            if use_collision_map:
                mapping_query += f"""Please use this collision diagram of the current game screen as a guide:
            
            {self.emulator.get_collision_map()}

            Legend:
            █ = Impassable
            · = Passable
            S - Sprite (NPC or Item or Misc)
            ↑ - Player (with Direction facing)
            | - edge of screen
            - - edge of screen

            Make sure to include a legend for whatever map you make.

            The top left corner of the map is ({coords[0] - 4}, {coords[1] - 4}). The bottom right is ({coords[0] + 5}, {coords[1] + 5}).

            Remember, higher numbers in the first coordinate are to the RIGHT. Higher numbers in the second coordinate are DOWN.
            """
        else: 
            mapping_query = f"""Current location: {location}
            
Here is the current map:

{this_map}
"""         
            if use_collision_map:
                mapping_query += f"""
Please use this collision diagram of the current game screen as a guide BUT REMEMBER to include the EXISTING MAP, which includes off-screen regions.
Make sure to modify the existing map unless we are in a menu or combat and the overworld isn't visible!
            
{self.emulator.get_collision_map()}

        Legend:
            █ = Impassable
            · = Passable
            S - Sprite (NPC or Item or Misc)
            ↑ - Player (with Direction facing)
            | - edge of screen
            - - edge of screen

Make sure to include a legend for whatever map you make.

The top left corner of the map is ({coords[0] - 4}, {coords[1] - 4}). The bottom right is ({coords[0] + 5}, {coords[1] + 5}).

Remember, higher numbers in the first coordinate are to the RIGHT. Higher numbers in the second coordinate are DOWN."""
        if MAPPING_MODEL == "CLAUDE": 
            if include_screenshot:
                messages = [{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": mapping_query,
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_b64,
                                },
                            },
                        ],
                    }]
            else:
                messages = [{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": mapping_query,
                            },
                        ],
                    }]
            # Get map from Claude
            response = self.anthropic_client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                system=MAPPING_PROMPT_SCREENSHOT if include_screenshot else MAPPING_PROMPT_NO_SCREENSHOT,
                messages=messages,
                temperature=TEMPERATURE
            )
            # Extract the new map
            full_text = " ".join([block.text for block in response.content if block.type == "text"])
        elif MAPPING_MODEL == "GEMINI":
            config=types.GenerateContentConfig(
                    max_output_tokens=None,
                    temperature=TEMPERATURE,
                    system_instruction=MAPPING_PROMPT_SCREENSHOT if include_screenshot else MAPPING_PROMPT_NO_SCREENSHOT,
                    tools=GOOGLE_TOOLS
                )
            chat = self.gemini_client.chats.create(
                model="gemini-2.5-pro-exp-03-25",
                config=config
            )   # context caching not available on gemini 2.5
            # catch/retry 500 errors
            retry_limit = 2
            cur_retries = 0
            while cur_retries < retry_limit:
                try:
                    if include_screenshot:
                        response = chat.send_message([types.Part(text=mapping_query), types.Part.from_bytes(data=screenshot_b64, mime_type="image/png")], config=config)
                    else:
                        response = chat.send_message([types.Part(text=mapping_query)], config=config)
                    break
                except ServerError as e:
                    if e.code != 500:
                        raise e
                    cur_retries += 1
            full_text = " ".join(
                x.text for x in response.candidates[0].content.parts if x.text is not None
            )
        elif MAPPING_MODEL == "OPENAI":
            if include_screenshot:
                messages = [{
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": mapping_query,
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{screenshot_b64}",
                            },
                        ],
                    }]
            else:
                messages = [{
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": mapping_query,
                            },
                        ],
                    }]
            response = self.openai_client.responses.create(
                model="o3",
                input=messages,
                instructions=MAPPING_PROMPT_SCREENSHOT if include_screenshot else MAPPING_PROMPT_NO_SCREENSHOT,
                max_output_tokens=MAX_TOKENS_OPENAI,
                temperature=TEMPERATURE,
                tools=OPENAI_TOOLS
            )

            response_texts = ""
            for chunk in response.output:
                if isinstance(chunk, responses.ResponseOutputMessage):
                    try:
                        response_texts += "\n".join(x.text for x in chunk.content)
                    except AttributeError:
                        # This was probably refused for safety reasons. Wait what?
                        breakpoint()
                else:
                    continue
            full_text = response_texts
            logger.info(f"Response usage: {response.usage.total_tokens if response.usage is not None else 'Unknown'}")
        else:
            raise ValueError("Unknown Mapping Model???")
        if "EXPLORATION COMMAND" in full_text:
            full_map, exploration_command = full_text.split("EXPLORATION COMMAND")
        else:
            exploration_command = None
            full_map = full_text
        self.map_tool_map[location] = full_map
        return exploration_command



    # Not really used anymore but was the original summarizer
    def summarize_history(self):
        """Generate a summary of the conversation history and replace the history with just the summary."""
        logger.info(f"[Agent] Generating conversation summary...")

        if MODEL == "CLAUDE":
            prompt_with_memory = f"""{SUMMARY_PROMPT_CLAUDE}

            Current Game State (Read directly from RAM! Do not question this!)

            {memory_state_string}
"""
        elif MODEL == "GEMINI":
            prompt_with_memory = f"""{SUMMARY_PROMPT_GEMINI}

            Current Game State (Read directly from RAM! Do not question this!)

            {memory_state_string}
"""
        elif MODEL == "OPENAI":
            prompt_with_memory = f"""{SUMMARY_PROMPT_OPENAI}

            Current Game State (Read directly from RAM! Do not question this!)

            {memory_state_string}
"""
        summary_text = self.prompt_text_reply(SYSTEM_PROMPT, prompt_with_memory, True, MODEL, False)
    
        self.text_display.add_message(f"[Agent] Game Progress Summary:")
        self.text_display.add_message(f"{summary_text}")
        
        # Get game state from memory after the action
        memory_state_string, location, coords = self.emulator.get_state_from_memory()


        # Get a fresh screenshot after executing the buttons
        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)

        # Replace message history with just the summary
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}"
                    },
                    {
                        "type": "text",
                        "text": "\n\nCurrent game screenshot for reference:"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "You were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next action."
                    },
                ]
            }
        ]
        self.openai_message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}" +
                                 "\n\nCurrent game screenshot is also included." +
                                 "\n\nYou were just asked to summarize your playthrough so far, which is the summary you see above. You may now continue playing by selecting your next a.")
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                    },
                ]
            }
        ]
        
        logger.info(f"[Agent] Message history condensed into summary.")