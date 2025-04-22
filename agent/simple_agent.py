import base64
import copy
import io
import json
import logging
import numpy as np
import os
import pickle
from PIL import ImageDraw, ImageFilter

from config import MAX_TOKENS, MODEL_NAME, TEMPERATURE
from agent.prompts import *
from secret_api_keys import *
from agent.emulator import Emulator
from agent.tool_definitions import AVAILABLE_TOOLS, GOOGLE_TOOLS, OPENAI_TOOLS

from anthropic import Anthropic
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from openai import OpenAI
from openai.types import responses
from openai import BadRequestError

from typing import Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_IMAGE_SIZE = (160, 144)  # In tiles, 16 x 10 columns and 16 x 9 rows

MODEL = "CLAUDE"
MAPPING_MODEL = "CLAUDE"
MAX_TOKENS_OPENAI = 50000

# Handles making an automatically updating collision map of an area as the model paths through it.
class LocationCollisionMap:
    def __init__(self, initial_collision_map: np.ndarray, initial_sprite_locations: set[tuple[int, int]], initial_coords: tuple[int, int]):
        # initial_collision_map is a 9 x 10 player-centered collision map which is 0 is impassable and 1 otherwise
        # Internally we store an expanding map based on locations we've been, with -1 in unknown spots, 2 in sprite locations, 3 in player location, and otherwise 0/1 as well.
        # We just accept that moving NPC locations are going to be inaccurate.
        # Note that while player coords are column, row, by default what we get from the collision map tooling is row, column
        self.player_coords = initial_coords
        self.col_offset = initial_coords[0] - 4
        self.row_offset = initial_coords[1] - 4
        self.internal_map = -np.ones((10, 9), dtype=np.int8) # We make our map the "proper" order that everything else is.
        for row in range(9):
            for col in range(10):
                if row == 4 and col == 4:  # player character
                    self.internal_map[col][row] = 3
                elif (col, row) in initial_sprite_locations:
                    self.internal_map[col][row] = 2
                else:
                    self.internal_map[col][row] = initial_collision_map[row][col]

    def update_map(self, collision_map: np.ndarray, sprite_locations: set[tuple[int, int]], coords: tuple[int, int]):
        # Remove the previous player marker. Most convenient to do it right now.
        self.internal_map[self.player_coords[0] - self.col_offset][self.player_coords[1] - self.row_offset] = 1
        self.player_coords = coords

        new_min_col = coords[0] - 4
        new_min_row = coords[1] - 4
        new_max_col = coords[0] + 5
        new_max_row = coords[1] + 4
        cur_size = self.internal_map.shape
        # First check if we need to move the boundaries of the array. Numpy pad makes this easy!
        expand_col_front = self.col_offset - new_min_col if new_min_col < self.col_offset else 0
        expand_col_back = new_max_col - (self.col_offset + cur_size[0] - 1) 
        expand_col_back = expand_col_back if expand_col_back > 0 else 0
        expand_row_front = self.row_offset - new_min_row if new_min_row < self.row_offset else 0
        expand_row_back = new_max_row - (self.row_offset + cur_size[1] - 1)
        expand_row_back = expand_row_back if expand_row_back > 0 else 0
        self.internal_map = np.pad(self.internal_map, pad_width=[(expand_col_front, expand_col_back), (expand_row_front, expand_row_back)], constant_values=-1)

        self.col_offset = min(new_min_col, self.col_offset)
        self.row_offset = min(new_min_row, self.row_offset)

        # Now update the map
        local_col_offset = new_min_col - self.col_offset
        local_row_offset = new_min_row - self.row_offset
        for row in range(9):
            for col in range(10):
                if row == 4 and col == 4:  # player character
                    self.internal_map[col + local_col_offset][row + local_row_offset] = 3
                    continue
                # if self.internal_map[col + local_col_offset][row + local_row_offset] != -1:
                    # continue  # No need to set, and if we do it's just going to lead to extra sprites for moving NPCs
                if (col, row) in sprite_locations:
                    self.internal_map[col + local_col_offset][row + local_row_offset] = 2
                else:
                    self.internal_map[col + local_col_offset][row + local_row_offset] = collision_map[row][col]

    def to_ascii(self) -> str:
        horizontal_border = "+" + "-" * self.internal_map.shape[0] + "+"
        lines = [f"({self.col_offset}, {self.row_offset})", horizontal_border]
        for this_row in self.internal_map.transpose():  # transposing makes printing easier
            row = "|"
            for col in this_row:
                if col == -1:
                    row += " "
                elif col == 0:
                    row += "█"
                elif col == 1:
                    row += "·"
                elif col == 2:
                    row += "S"
                elif col == 3:
                    row += "P"
            row += "|"
            lines.append(row)
        lines.append(horizontal_border + f"({self.col_offset + self.internal_map.shape[0] - 1}, {self.row_offset + self.internal_map.shape[1] - 1})")

        # Add legend
        lines.extend(
            [
                "",
                "Legend:",
                "█ - Wall/Obstacle",
                "· - Path/Walkable",
                "S - Sprite",
                "P - Player Character",
                "  - Blank = Unknown/Unvisited"
            ]
        )

        # Join all lines with newlines
        output = "\n".join(lines)
        with open("mapping_log.txt", "w", encoding="utf-8") as fw:
            fw.write(output)
        return output

# Updates a text file over time to write the last X blocks of text to a text file so we can see it well
# with tail -F or something.
# TODO: This should definitely be a display buffer or something, not 100000 writes to the hard disk.
class TextDisplay:
    FILE_NAME = "text_output.txt"

    def __init__(self, message_depth=20):
        self.text_buffer = []
        self.message_depth = message_depth

    def add_message(self, message: str):
        self.text_buffer.append(message)
        logger.info(message)
        if len(self.text_buffer) > self.message_depth:
            self.text_buffer = self.text_buffer[1:]
        with open(self.FILE_NAME, "w", encoding="utf-8") as fw:
            fw.write("\n\n".join(self.text_buffer))


class SimpleAgent:
    def __init__(self, rom_path, headless=True, sound=False, max_history=60, load_state=None, location_history_length=40, location_archive_file_name: Optional[str]=None, use_full_collision_map: bool=True):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
        """
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()  # Initialize the emulator
        if MODEL == "CLAUDE" or MAPPING_MODEL == "CLAUDE":
            self.anthropic_client = Anthropic(api_key=API_CLAUDE, max_retries=10)
        if MODEL == "GEMINI" or MAPPING_MODEL == "GEMINI":
            self.gemini_client = genai.Client(api_key=API_GOOGLE)
        if MODEL == "OPENAI" or MAPPING_MODEL == "OPENAI":
            self.openai_client = OpenAI(api_key=API_OPENAI)
        self.running = True
        # TODO: OKAY LOOK this was originally a pretty small state and that it got out of hand.
        self.message_history = [{"role": "user", "content": "You may now begin playing."}]
        self.openai_message_history = [{"role": "user", "content": "You may now begin playing."}]
        self.max_history = max_history
        self.location_history_length = location_history_length
        self.location_archive_file_name = location_archive_file_name
        self.location_history: list[tuple[str, tuple[int, int]]] = []
        # location -> row -> col -> label. By nesting the dicts we make the lookup
        # faster, but this is a crappy way of doing it.
        self.label_archive: dict[str, dict[int, dict[int, str]]] = {}
        # the dedicated location tracker that the model may turn on
        self.location_tracker_activated: bool = False
        self.location_tracker: dict[str, list[list[bool]]] = {}  # True if visited, False if not, could be numpy but eh for now.
        self.steps_since_checkpoint = 0
        self.steps_since_label_reset = 0
        self.last_location: Optional[str] = None
        self.map_tool_map: dict[str, str] = {}  # location -> map
        self.fully_mapped_locations: set[str] = set()  # Unused for now
        self.full_collision_map: dict[str, LocationCollisionMap] = {}
        self.use_full_collision_map = use_full_collision_map  # Do I need to save this?
        self.absolute_step_count = 0
        self.all_visited_locations: set[str] = set()
        self.location_milestones: list[tuple[str, int]] = []
        self.text_display = TextDisplay()
        self.last_coords = None  # A bit more precise, since it includes detailed trajectories from push button and navigate to.
        self.checkpoints = []  # A long-running list of achievements, used to track internal progres.

        if load_state:
            logger.info(f"Loading saved state from {load_state}")
            self.emulator.load_state(load_state)
            self.load_location_archive(location_archive_file_name)

    # This does the overlay for the model.
    def get_screenshot_base64(
            self, screenshot, upscale=1, add_coords: bool=True,
            player_coords: Optional[tuple[int, int]]=None, location: Optional[str]=None, relative_square_size=8):
        """Convert PIL image to base64 string."""
        # Resize if needed
        if upscale > 1:
            new_size = (screenshot.width * upscale, screenshot.height * upscale)
            screenshot = screenshot.resize(new_size)

        past_locations = self.location_history
        location_labels = self.label_archive.get(location)
        if location_labels is None:
            # this sucks man
            for key, value in self.label_archive.items():
                if location.lower() == key.lower():
                    location_labels = value
                    break
        if location_labels is None:
            location_labels = {}
        local_location_tracker = self.location_tracker.get(location, [])

        collision_map = self.emulator.pyboy.game_wrapper.game_area_collision()
        downsampled_terrain = self.emulator._downsample_array(collision_map)

        sprite_locations = self.emulator.get_sprites()

        if not self.emulator.get_in_combat():
            # add coordinate labels (note: if scale is too small it may be unreadable)
            # The assumption is the central square is the player's current location, which is 4, 4
            # Rows 0 - 8, Cols 0 - 9
            if add_coords:
                assert player_coords is not None
                tile_size = 16 * upscale
                mid_length = tile_size/2
                for row in range(0, 9):
                    # For bad legacy reasons location labels is row first
                    real_row = player_coords[1] + row - 4
                    local_cols = location_labels.get(real_row, {})
                    for col in range(0, 10):
                        if row == 4 and col == 4:
                            continue  # Skip the player themselves.
                        real_col = player_coords[0] + col - 4
                        label = local_cols.get(real_col, "")
                        tile_label = f"{str(real_col)}, {str(real_row)}"
                        if label:
                            tile_label += "\n" + label
                        if (col, row) not in sprite_locations:
                            if downsampled_terrain[row][col] == 0:
                                # ImageDraw.Draw(screenshot).rectangle(((col * tile_size + (relative_square_size - 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size - 1)*mid_length/relative_square_size), (col * tile_size + (relative_square_size + 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size + 1)*mid_length/relative_square_size)), (255, 0, 0))
                                tile_label += "\n" + "IMPASSABLE"
                            else:
                                # ImageDraw.Draw(screenshot).rectangle(((col * tile_size + (relative_square_size - 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size - 1)*mid_length/relative_square_size), (col * tile_size + (relative_square_size + 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size + 1)*mid_length/relative_square_size)), (0, 255, 255))
                                if local_location_tracker and real_col > -1 and real_row > -1 and real_col < len(local_location_tracker) and real_row < len(local_location_tracker[real_col]) and local_location_tracker[real_col][real_row]:
                                    # ImageDraw.Draw(screenshot).rectangle(((col * tile_size + (relative_square_size - 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size - 1)*mid_length/relative_square_size), (col * tile_size + (relative_square_size + 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size + 1)*mid_length/relative_square_size)), (0, 0, 255))
                                    tile_label += "\n" + "EXPLORED"
                                elif (location, (real_col, real_row)) in past_locations:
                                    # ImageDraw.Draw(screenshot).rectangle(((col * tile_size + (relative_square_size - 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size - 1)*mid_length/relative_square_size), (col * tile_size + (relative_square_size + 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size + 1)*mid_length/relative_square_size)), (0, 255, 0))         
                                    tile_label += "\n" + "RECENTLY\nVISITED"
                                else:
                                    tile_label += "\n" + "CHECK\nHERE"
                        else:
                            # ImageDraw.Draw(screenshot).rectangle(((col * tile_size + (relative_square_size - 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size - 1)*mid_length/relative_square_size), (col * tile_size + (relative_square_size + 1)*mid_length/relative_square_size, row * tile_size + (relative_square_size + 1)*mid_length/relative_square_size)), (255, 0, 255))
                            tile_label += "\n" + "NPC/OBJECT"
                        font_size = 8
                        if MODEL == "GEMINI":
                            font_size = 12
                        ImageDraw.Draw(screenshot).text((col * tile_size + mid_length/2, row * tile_size + mid_length/2), tile_label, (255, 0, 0), font_size=font_size)
        screenshot.save("test.png")  # expensive, remove later

        # Convert to base64
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        return base64.standard_b64encode(buffered.getvalue()).decode()

    # Important for maintaining state between runs.
    # I am aware that this has grown into a list of like 17 pickle dumps. It got out of hand.
    def save_location_archive(self, pkl_path: str) -> None:
        # TODO: I should really just make this a clear state variable.
        with open(pkl_path, 'wb') as fw:
            pickle.dump(self.label_archive, fw)
            pickle.dump(self.location_history, fw)
            pickle.dump(self.message_history, fw)
            pickle.dump(self.location_tracker, fw)
            pickle.dump(self.location_tracker_activated, fw)
            pickle.dump(self.steps_since_checkpoint, fw)
            pickle.dump(self.steps_since_label_reset, fw)
            pickle.dump(self.last_location, fw)
            pickle.dump(self.map_tool_map, fw)
            pickle.dump(self.fully_mapped_locations, fw)
            pickle.dump(self.openai_message_history, fw)
            pickle.dump(self.full_collision_map, fw)
            pickle.dump(self.absolute_step_count, fw)
            pickle.dump(self.all_visited_locations, fw)
            pickle.dump(self.location_milestones, fw)
            pickle.dump(self.last_coords, fw)
            pickle.dump(self.checkpoints, fw)

    def load_location_archive(self, pkl_path: str) -> None:
        try:
            with open(pkl_path, 'rb') as fr:
                self.label_archive = pickle.load(fr)
                self.location_history = pickle.load(fr)
                self.message_history = pickle.load(fr)
                try:  # temporary legacy: older pickles don't have this
                    self.location_tracker = pickle.load(fr)
                    self.location_tracker_activated = pickle.load(fr)
                    self.steps_since_checkpoint = pickle.load(fr)
                    self.steps_since_label_reset = pickle.load(fr)
                    self.last_location = pickle.load(fr)
                    self.map_tool_map = pickle.load(fr)
                    self.fully_mapped_locations = pickle.load(fr)
                    self.openai_message_history = pickle.load(fr)
                    self.full_collision_map = pickle.load(fr)
                    self.absolute_step_count = pickle.load(fr)
                    self.all_visited_locations = pickle.load(fr)
                    self.location_milestones = pickle.load(fr)
                    self.last_coords = pickle.load(fr)
                    self.checkpoints = pickle.load(fr)
                except Exception:
                    pass
        except FileNotFoundError:
            logger.warn("No Location archive! Making new one...")
        if not isinstance(self.message_history[-1]['content'], str):
            for entry in self.message_history[-1]['content']:
                if entry['type'] == 'tool_use':
                    self.message_history.pop()
                    break
        # TODO: Code for being able to restart gemini/openai with an interrupted tool call. Currently may glitch out in that scenario.

    def update_and_get_full_collision_map(self, location, coords):
        collision_map = self.emulator.pyboy.game_wrapper.game_area_collision()
        downsampled_terrain = self.emulator._downsample_array(collision_map)
        # slightly more efficient than setdefault
        this_map = self.full_collision_map.get(location)
        if this_map is None:
            self.full_collision_map[location] = LocationCollisionMap(downsampled_terrain, self.emulator.get_sprites(), coords)
            return self.full_collision_map[location].to_ascii()
        else:
            this_map.update_map(downsampled_terrain, self.emulator.get_sprites(), coords)
            return this_map.to_ascii()

    # TODO: An obvious refactor would be to move some of these into their own functions.
    def process_tool_call(self, tool_call):
        """Process a single tool call."""
        tool_name = tool_call.name
        if MODEL == "CLAUDE":
            tool_input = tool_call.input
            tool_id = tool_call.id
        elif MODEL == "GEMINI":
            tool_input = tool_call.args
            tool_id = tool_call.id
        elif MODEL == "OPENAI":
            tool_input = json.loads(tool_call.arguments)
            tool_id = tool_call.call_id
            self.text_display.add_message(f"[Text] {tool_input['explanation_of_action']}")
        logger.info(f"Processing tool call: {tool_name}")

        if tool_name == "press_buttons":
            buttons = tool_input["buttons"]
            wait = tool_input.get("wait", True)
            self.text_display.add_message(f"[Buttons] Pressing: {buttons} (wait={wait})")
            
            result, last_coords = self.emulator.press_buttons(buttons, wait)
            
            self.last_coords = last_coords
            
            # Get game state from memory after the action
            memory_info, location, coords = self.emulator.get_state_from_memory()
            
            # Log the memory state after the tool call
            logger.info(f"[Memory State after action]")
            logger.info(memory_info)
            
            collision_map = self.emulator.get_collision_map()
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")

            # TODO: Maybe python has good queues for this, but queue is not iterable for display
            self.location_history.insert(0, (location, coords))
            if len(self.location_history) > self.location_history_length:
                self.location_history.pop()
            if self.location_tracker_activated:
                cols = self.location_tracker.setdefault(location, [])
                # This is leaning hard on Python list append optimization... Maybe there are better structures?
                # col first
                if coords[0] > len(cols) - 1:
                    if len(cols) == 0:
                        cols.extend(list() for _ in range(0, coords[0] + 1))  # Note that you can't do []*coords[0], because then the same list goes into each entry
                    else:
                        cols.extend([False for _ in range(0, len(cols[0]))] for _ in range(0, coords[0] + 1))
                if coords[1] > len(cols[0]) - 1:
                    # this is awkward
                    for col in cols:
                        # This is actually too much (it would be coords[1] - len(col) + 1) but the overallocation is probably a good idea.
                        col.extend(False for _ in range(0, coords[1] + 1))
                cols[coords[0]][coords[1]] = True

            # TODO: eventually do this more reasonably. For now we do this extraordinarily dumb approach.
            col = coords[0]
            row = coords[1]
            all_labels = []
            this_location = self.label_archive.get(location)
            if this_location is None:
                # this sucks man
                for key, value in self.label_archive.items():
                    if location.lower() == key.lower():
                        this_location = value
                        break
            if this_location is not None:
                for nearby_row in range(max(0, row-10), row+11):
                    this_row = this_location.get(nearby_row)
                    if this_row is not None:
                        for nearby_col in range(max(0, col-10), col+11):
                            this_col = this_row.get(nearby_col)
                            if this_col is not None:
                                all_labels.append(((nearby_col, nearby_row), this_col))  # Note that we only care about our current location


            # Return tool result as a dictionary
            # OPENAI doesn't know what to do with too much information here.
            if MODEL == "OPENAI":
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
                # Get a fresh screenshot after executing the buttons
                screenshot = self.emulator.get_screenshot()
                screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                last_checkpoints = '\n'.join(self.checkpoints[-10:])
                content = [
                        {"type": "text", "text": f"Pressed buttons: {', '.join(buttons)}"},
                        {"type": "text", "text": "\nHere is a screenshot of the screen after your button presses:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64,
                            },
                        },
                        {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                        {"type": "text", "text": f"\nLabeled nearby locations: {','.join(f'{label_coords}: {label}' for label_coords, label in all_labels)}"},
                        {"type": "text", "text": f"Here are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}"},
                        {"type": "text", "text": f"Here are your last 10 checkpoints:\n{last_checkpoints}"}
                    ]
                if not self.emulator.get_in_combat() and self.use_full_collision_map:
                    content.append({"type": "text", "text": "Here is an ASCII map of this RAM location compiled so far:\n\n" + self.update_and_get_full_collision_map(location, coords)})
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                }
        elif tool_name == "navigate_to":
            row = tool_input["row"]
            col = tool_input["col"]
            memory_info, location, coords = self.emulator.get_state_from_memory()
            self.text_display.add_message(f"[Navigation] Navigating to: ({col}, {row})")  # 8, 3 -> 6, 4 is 2, 5
            
            # The navigator goes to location on screen, with 0,0 at the top left.
            local_col = col - coords[0] + 4
            local_row = row - coords[1] + 4

            status, path = self.emulator.find_path(local_row, local_col)
            last_coords = coords
            next_coords = coords
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True)
                    cur_coords = self.emulator.get_coordinates()
                    if cur_coords != next_coords:
                        last_coords = next_coords
                        next_coords = cur_coords
                self.last_coords = last_coords
                result = f"Navigation successful: followed path with {len(path)} steps"
            else:
                result = f"Navigation failed: {status}"
            
            # Get game state from memory after the action
            memory_info, location, coords = self.emulator.get_state_from_memory()

            # Get a fresh screenshot after executing the buttons
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
            
            # Log the memory state after the tool call
            logger.info(f"[Memory State after action]")
            logger.info(memory_info)
            
            collision_map = self.emulator.get_collision_map()
            if collision_map:
                logger.info(f"[Collision Map after action]\n{collision_map}")

            # TODO: Maybe python has good queues for this, but queue is not iterable for display
            self.location_history.insert(0, (location, coords))
            if len(self.location_history) > self.location_history_length:
                self.location_history.pop()
            if self.location_tracker_activated:
                if coords[0] >= 0 or coords[1] >= 0:
                    try:
                        # Covers edge cases, principally when moving between areas.
                        cols = self.location_tracker.setdefault(location, [])
                        # This is leaning hard on Python list append optimization... Maybe there are better structures?
                        # col first
                        if coords[0] > len(cols) - 1:
                            if len(cols) == 0:
                                cols.extend(list() for _ in range(0, coords[0] + 1))  # Note that you can't do []*coords[0], because then the same list goes into each entry
                            else:
                                cols.extend([False for _ in range(0, len(cols[0]))] for _ in range(0, coords[0]))
                        if coords[1] > len(cols[0]) - 1:
                            # this is awkward
                            for col in cols:
                                # This is actually too much (it would be coords[1] - len(col) + 1) but the overallocation is probably a good idea.
                                col.extend(False for _ in range(0, coords[1] + 1))
                        cols[coords[0]][coords[1]] = True
                    except IndexError as e:
                        # I may have fixed this error, but leaving this here in case of debugging. Can remove eventually.
                        breakpoint()

            # TODO: eventually do this more reasonably. For now we do this extraordinarily dumb approach.
            col = coords[0]
            row = coords[1]
            all_labels = []
            this_location = self.label_archive.get(location)
            if this_location is None:
                # this sucks man
                for key, value in self.label_archive.items():
                    if location.lower() == key.lower():
                        this_location = value
                        break
            if this_location is not None:
                for nearby_row in range(max(0, row-10), row+11):
                    this_row = this_location.get(nearby_row)
                    if this_row is not None:
                        for nearby_col in range(max(0, col-10), col+11):
                            this_col = this_row.get(nearby_col)
                            if this_col is not None:
                                all_labels.append(((nearby_col, nearby_row), this_col))  # Note that we only care about our current location

            # Return tool result as a dictionary
            # OPENAI doesn't know what to do with too much information here.
            if MODEL == "OPENAI":
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {"type": "text", "text": (
                            f"Navigation result: {result}"
                        )}
                    ],
                }
            else:
                # Get a fresh screenshot after executing the buttons
                screenshot = self.emulator.get_screenshot()
                screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                last_checkpoints = '\n'.join(self.checkpoints[-10:])
                content = [
                        {"type": "text", "text": f"Navigation result: {result}"},
                        {"type": "text", "text": "\nHere is a screenshot of the screen after navigation:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_b64,
                            },
                        },
                        {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                        {"type": "text", "text": f"\nLabeled nearby locations: {','.join(f'{coords}: {label}' for coords, label in all_labels)}"},
                        {"type": "text", "text": f"Here are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}"},
                        {"type": "text", "text": f"Here are your last 10 checkpoints:\n{last_checkpoints}"}
                    ]
                if not self.emulator.get_in_combat() and self.use_full_collision_map:
                    content.append({"type": "text", "text": "Here is an ASCII map of this RAM location compiled so far:\n\n" + self.update_and_get_full_collision_map(location, coords)})
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                }
        elif tool_name == "bookmark_location_or_overwrite_label":
            location = tool_input["location"]
            row = tool_input["row"]
            col = tool_input["col"]
            label = tool_input["label"]
            self.text_display.add_message(f"Logging {location},  ({col}, {row}) as {label}")
            self.label_archive.setdefault(location.lower(), {}).setdefault(row, {})[col] = label
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": f"Location Labeled: {location}, ({col}, {row}) as {label}"}
                ],
            }
        elif tool_name == "mark_checkpoint":
            self.steps_since_checkpoint = 0
            self.steps_since_label_reset = 0
            self.location_tracker_activated = False
            achievement = tool_input["achievement"]
            self.checkpoints.append(achievement)
            self.text_display.add_message(f"Checkpoint marked: {achievement}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": f"Checkpoint set!"}
                ],
            }
        elif tool_name == "navigation_assistance":
            assist_str = self.navigation_assistance(tool_input["navigation_goal"])
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": assist_str}
                ],
            }
        else:
            logger.error(f"Unknown tool called: {tool_name}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}
                ],
            }

    def run(self, num_steps=1, save_every=10, save_file_name: Optional[str] = None):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """
        logger.info(f"Starting agent loop for {num_steps} steps")

        steps_completed = 0
        last_location = None
        while self.running and steps_completed < num_steps:
            try:
                location = self.emulator.get_location()
                coords = self.emulator.get_coordinates()
                if location not in self.all_visited_locations:
                    self.text_display.add_message(f"New Location reached! {location} at {self.absolute_step_count}")
                    self.location_milestones.append((location, self.absolute_step_count))
                    self.all_visited_locations.add(location)
                if location != last_location:
                    if self.last_coords is not None and not self.emulator.get_in_combat():
                        self.label_archive.setdefault(last_location, {}).setdefault(self.last_coords[1], {})[self.last_coords[0]] = f"Entrance to {location} (Approximate)"
                    last_location = location
                self.last_coords = coords
                malformed = False
                messages = copy.deepcopy(self.message_history)

                if len(messages) >= 3:
                    if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
                    
                    if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                        messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

                token_usage = 0

                # Get model response
                if MODEL == "CLAUDE":
                    response = self.anthropic_client.messages.create(
                        model=MODEL_NAME,
                        max_tokens=MAX_TOKENS,
                        system=SYSTEM_PROMPT,
                        messages=messages,
                        tools=AVAILABLE_TOOLS,
                        temperature=TEMPERATURE,
                    )
                    token_usage = response.usage.input_tokens + response.usage.output_tokens
                    logger.info(f"Response usage: {response.usage}")
                    # Extract tool calls
                    tool_calls = [
                        block for block in response.content if block.type == "tool_use"
                    ]

                    # Display the model's reasoning
                    for block in response.content:
                        if block.type == "text":
                            self.text_display.add_message(f"[Text] {block.text}")
                        elif block.type == "tool_use":
                            self.text_display.add_message(f"[Tool] Using tool: {block.name}")

                
                    # Process tool calls
                    if tool_calls:
                        # Add assistant message to history
                        assistant_content = []
                        for block in response.content:
                            if block.type == "text":
                                assistant_content.append({"type": "text", "text": block.text})
                            elif block.type == "tool_use":
                                assistant_content.append({"type": "tool_use", **dict(block)})
                elif MODEL == "GEMINI":
                    # messages -> Gemini format
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
                                    all_text = [content['content'][0]['text']]
                                    if len(content['content']) > 1:
                                        all_text.append(content['content'][1]['text'])
                                        for x in range(3, 6):
                                            all_text.append(content['content'][x]['text'])
                                    all_text = "\n".join(all_text)
                                    response_dict = copy.copy(content['content'][0])
                                    response_dict['text'] = all_text
                                    parts.append(types.Part.from_function_response(name=last_tool_used, response=response_dict))

                                    if len(content['content']) > 1:
                                        parts.append(types.Part.from_bytes(data=content['content'][2]['source']['data'], mime_type=content['content'][2]['source']['media_type']))
                                        last_image_part = parts[-1]
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

                    config=types.GenerateContentConfig(
                            max_output_tokens=None,
                            temperature=TEMPERATURE,
                            system_instruction=SYSTEM_PROMPT,
                            tools=GOOGLE_TOOLS
                        )
                    chat = self.gemini_client.chats.create(
                        model="gemini-2.5-flash-preview-04-17",
                        history=google_messages[:-1],
                        config=config
                    )   # context caching not available on gemini 2.5
                    retry_limit = 2
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
                    logger.info(f"Response usage: {response.usage_metadata.total_token_count}")
                    tool_calls = []
                    assistant_content = []
                    malformed = False
                    if response.candidates is not None:
                        try:
                            for part in response.candidates[0].content.parts:
                                if part.text is not None:
                                    # Gemini is inconsistent about tool use and will sometimes slip into pure text mode
                                    # inexplicably. If this happens, accommodate it but also berate it for not replyig
                                    # properly.
                                    if part.text.startswith("```python"):
                                        logging.info("[Malformed reply?] Attempting to parse anyway...")
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
                                        self.text_display.add_message(f"[Text] {part.text}")
                                        assistant_content.append({"type": "text", "text": part.text})
                                if part.function_call is not None:
                                    logger.info(f"[Tool] Using tool: {part.function_call.name}")
                                    assistant_content.append({"type": "tool_use", "id": part.function_call.id, "input": part.function_call.args, "name": part.function_call.name})
                                    tool_calls.append(part.function_call)
                        except Exception as e:
                            print(e)
                            breakpoint()
                        token_usage = 0  # I didn't even remember to track this but it probably doesn't matter
                elif MODEL == "OPENAI":
                    # For openai we need to add a screenschot too to tool calls or it gets very confused.
                    # Get a fresh screenshot after executing the buttons
                    if isinstance(self.openai_message_history[-1], dict) and self.openai_message_history[-1].get('type') == "function_call_output":
                        # Unfortunately this is buried...
                        # parsed_result = json.loads(self.openai_message_history[-1]["output"])
                        # Apparently openai can get confused without a fresh update.
                        #if parsed_result[0]["text"].startswith("Pressed buttons") or parsed_result[0]["text"].startswith("Navigation"):
                        memory_info, location, coords = self.emulator.get_state_from_memory()
                        # TODO: eventually do this more reasonably. For now we do this extraordinarily dumb approach.
                        col = coords[0]
                        row = coords[1]
                        all_labels: list[tuple[tuple[int, int], str]] = []
                        this_location = self.label_archive.get(location)
                        if this_location is None:
                            # this sucks man
                            for key, value in self.label_archive.items():
                                if location.lower() == key.lower():
                                    this_location = value
                                    break
                        if this_location is not None:
                            for nearby_row in range(max(0, row-10), row+11):
                                this_row = this_location.get(nearby_row)
                                if this_row is not None:
                                    for nearby_col in range(max(0, col-10), col+11):
                                        this_col = this_row.get(nearby_col)
                                        if this_col is not None:
                                            all_labels.append(((nearby_col, nearby_row), this_col))  # Note that we only care about our current location
                        screenshot = self.emulator.get_screenshot()
                        screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                        last_checkpoints = '\n'.join(self.checkpoints[-10:])
                        content = [
                                {
                                    "type": "input_text",
                                    "text": (f"\nGame state information from memory after your action:\n{memory_info}"
                                            f"\nLabeled nearby locations: {','.join(f'{label_coords}: {label}' for label_coords, label in all_labels)}" +
                                            f"Here are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}" +
                                            f"Here are your last 10 checkpoints:\n{last_checkpoints}"),
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{screenshot_b64}",
                                },
                            ]
                        if not self.emulator.get_in_combat() and self.use_full_collision_map:
                            content[0]['text'] += "\n\nHere is an ASCII map of this RAM location compiled so far:\n\n" + self.update_and_get_full_collision_map(location, coords)
                        self.openai_message_history.append({
                            "role": "user",
                            "content": content,  # type: ignore
                        })
                    retries = 2
                    cur_tries = 0
                    while cur_tries < retries:
                        try:
                            response = self.openai_client.responses.create(
                                model="o3",
                                input=self.openai_message_history,  # type: ignore
                                instructions=SYSTEM_PROMPT_OPENAI,
                                max_output_tokens=MAX_TOKENS_OPENAI,
                                temperature=TEMPERATURE,
                                tools=OPENAI_TOOLS
                            )
                            break
                        except BadRequestError as e:
                            cur_tries += 1  # Sometimes it spuriously flags this as content violation. I don't know why.
                            continue
                        except Exception as e:
                            print(e)
                            breakpoint()
                    # We immediately drop the previous images because of resource costs (and context explosion)
                    for message in self.openai_message_history:
                        # here we exploit the fact that it's always a dict if we're putting in images...
                        if isinstance(message, dict):
                            try:
                                outputs = message['content'] if 'content' in message else message['output']
                                if isinstance(outputs, str):
                                    if 'output' in message:
                                        try:
                                            decoded = json.loads(outputs)
                                            for i, chunk in enumerate(decoded):
                                                if not isinstance(chunk, dict):
                                                    continue
                                                if chunk["type"] == "image":
                                                    decoded[i] = {
                                                            "type": "input_text",
                                                            "text": "Screenshot omitted in history for brevity",
                                                    }
                                            message['output'] = json.dumps(decoded)
                                        except json.JSONDecodeError:
                                            pass
                                else:
                                    for chunk in outputs:
                                        if not isinstance(chunk, dict):
                                            continue
                                        if chunk["type"] == "input_image":
                                            del chunk["image_url"]
                                            chunk["type"] = "input_text"
                                            chunk["text"] = "Screenshot omitted in history for brevity"
                            except KeyError as e:
                                print(e)
                                breakpoint()
                    self.openai_message_history.extend(response.output)  # type: ignore
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
                            self.text_display.add_message(f"[Text] {full_text}")
                            assistant_content.append({"type": "text", "text": full_text})
                    logger.info(f"Response usage: {response.usage.total_tokens if response.usage is not None else 'Unknown'}")
                    token_usage = response.usage.total_tokens if response.usage is not None else 0
                    
                
                # Process tool calls
                if tool_calls: 
                    self.message_history.append(
                        {"role": "assistant", "content": assistant_content}
                    )
                    
                    # Process tool calls and create tool results
                    tool_results = []
                    for tool_call in tool_calls:
                        tool_result = self.process_tool_call(tool_call)
                        tool_results.append(tool_result)
                        openai_result = {
                            "type": "function_call_output",
                            "call_id": tool_result["tool_use_id"],
                            "output": json.dumps(tool_result["content"])
                        }
                        self.openai_message_history.append(openai_result)

                    """# Call the mapping tool if not in combat, about every 10 actions.
                    if not self.use_full_collision_map and not self.emulator.get_in_combat() and (not steps_completed % 10):
                        location = self.emulator.get_location()
                        exploration_command = self.mapping_tool()
                        local_map = self.map_tool_map.get(
                            location
                        )
                        if local_map is not None:
                            tool_results.append(
                                {"type": "text", "text": f"HERE IS AN ASCII MAP OF THIS LOCATION BASED ON YOUR EXPLORATION: {local_map}"})
                        if exploration_command is not None:
                            tool_results.append(
                                {"type": "text", "text": f"EXPLORATION COMMAND: {exploration_command}"})
                            logger.info(f"Mapping tool forcing redirection: {exploration_command}")
                        with open("mapping_log.txt", "w", encoding="utf-8") as fw:
                            fw.write(local_map)"""
                    if malformed:
                        tool_results.append({"type": "text", "text": f"WARNING: MALFORMED TOOL CALL. Call using the function call, not in the text."})

                    # Add tool results to message history
                    self.message_history.append(
                        {"role": "user", "content": tool_results}  # type: ignore
                    )
                    # Check if we need to summarize the history
                    if len(self.message_history) >= self.max_history or (MODEL == "OPENAI" and token_usage > 170000):  # To my surprise, o3 runs out fasssst
                        self.agentic_summary()
                if not tool_calls:  # type: ignore
                    # Sometimes it just stalls out mysteriously or says some text.
                    self.message_history.append(
                        {"role": "user", "content": [{"text": "Can you please continue playing the game?", "type": "text"}]}  # type: ignore
                    )

                steps_completed += 1
                self.absolute_step_count += 1
                self.steps_since_checkpoint += 1
                self.steps_since_label_reset += 1
                if self.steps_since_checkpoint > 50 and not self.location_tracker_activated:
                    self.location_tracker_activated = True
                    self.location_tracker = {}
                _, location, _ = self.emulator.get_state_from_memory()
                if self.last_location != location:
                    self.steps_since_label_reset = 0
                self.last_location = location
                if self.steps_since_label_reset > (200 if MODEL == "CLAUDE" else 1000):
                    self.text_display.add_message("Clearing labels to clear potential bad labels...")
                    self.steps_since_label_reset = 0
                    self.label_archive[location] = {}
                logger.info(f"Completed step {steps_completed}/{num_steps}")
                self.text_display.add_message(f"Absolute step count: {self.absolute_step_count}")
                if save_file_name is not None and not steps_completed % save_every:
                    self.emulator.pyboy.save_state(open(save_file_name, "wb"))
                    self.save_location_archive(self.location_archive_file_name)
                    with open("location_milestones.txt", "w") as fw:
                        fw.write(str(self.location_milestones))

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                raise e
            
        if save_file_name is not None:
            logger.info("Saving state")
            self.emulator.pyboy.save_state(open(save_file_name, "wb"))
            self.save_location_archive(self.location_archive_file_name)
            with open("location_milestones.txt", "w") as fw:
                fw.write(str(self.location_milestones))

        if not self.running:
            self.emulator.stop()

        return steps_completed

    def navigation_assistance(self, navigation_goal: str) -> str:
        logger.info(f"[Agent] Running Navigation Assist...")
        
        _, location, coords = self.emulator.get_state_from_memory()

        collision_map = self.update_and_get_full_collision_map(location, coords)

        this_location = self.label_archive.get(location)
        if this_location is None:
            # this sucks man
            for key, value in self.label_archive.items():
                if location.lower() == key.lower():
                    this_location = value
                    break

        labels = "No Labels yet."
        all_labels = []
        if this_location is not None:
            for row_ind, this_row in this_location.items():
                for col_ind, this_col in this_row.items():
                    all_labels.append(((col_ind, row_ind), this_col)) 

        if all_labels:                
            labels = ','.join(f'{label_coords}: {label}' for label_coords, label in all_labels)

        mapping_query = f"""Here is a map of the current location:

        Current location: {location}

        {collision_map}

        Remember, higher numbers in the first coordinate are to the RIGHT. Higher numbers in the second coordinate are DOWN.
        
        Here are some labels:

        {labels}

        Here is the current navigation goal:

        {navigation_goal}
        """

        full_text = self.prompt_text_reply(NAVIGATION_PROMPT, mapping_query, False, MAPPING_MODEL, False)
        self.text_display.add_message(f"Navigation Advice: {full_text}")
        return full_text
    
    def prompt_text_reply(self, instructions: str, prompt: str, include_history: bool, model: str, include_screenshot: bool) -> str:

        if include_history:
            # Create messages for the summarization request - pass the entire conversation history
            messages = copy.deepcopy(self.message_history) 


            if len(messages) >= 3:
                if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                    messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
                
                if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                    messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        else:
            messages = []

        if include_screenshot:
            _, location, coords = self.emulator.get_state_from_memory()
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)

        if model == "CLAUDE":
            if include_screenshot:
                 messages += [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_b64,
                                },
                            }
                        ],
                    }, 
                ]
            else:
                messages += [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ]
            # Get text from Claude
            response = self.anthropic_client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                system=instructions,
                messages=messages,
                temperature=TEMPERATURE,
                thinking={"type": "enabled", "budget_tokens": 6000}
            )
            # Extract the text
            summary_text = " ".join([block.text for block in response.content if block.type == "text"])
        elif model == "GEMINI":
            if include_history:
                # messages -> Gemini format
                google_messages = []
                last_tool_used = None
                for i, message in enumerate(messages):
                    role = message["role"]
                    if role =="assistant":
                        role = "model"
                    parts = []
                    for content in message["content"]:
                        if isinstance(content, str):
                            parts.append(types.Part.from_text(text=content))
                        else:
                            if content['type'] == 'tool_result':
                                # gemini gets confused sometimes with too many text parts
                                all_text = [content['content'][0]['text']]
                                if len(content['content']) > 1:
                                    all_text.append(content['content'][1]['text'])
                                    for x in range(3, 6):
                                        all_text.append(content['content'][x]['text'])
                                response_dict = copy.copy(content['content'][0])
                                response_dict['text'] = all_text
                                parts.append(types.Part.from_function_response(name=last_tool_used, response=response_dict))

                                if len(content['content']) > 1:
                                    parts.append(types.Part.from_bytes(data=content['content'][2]['source']['data'], mime_type=content['content'][2]['source']['media_type']))
                            elif content['type'] == "text":
                                parts.append(types.Part.from_text(text=content['text']))
                            elif content['type'] == "image":
                                parts.append(types.Part.from_bytes(data=content['source']['data'], mime_type=content['source']['media_type']))
                            elif content['type'] == 'tool_use':
                                last_tool_used = content['name']
                                parts.append(types.Part.from_function_call(name=last_tool_used, args=content['input']))
                    google_messages.append(types.Content(parts=parts, role=role))
            else:
                google_messages = []

            config=types.GenerateContentConfig(
                    max_output_tokens=None,
                    temperature=TEMPERATURE,
                    system_instruction=instructions,
                    tools=GOOGLE_TOOLS
                )
            chat = self.gemini_client.chats.create(
                model="gemini-2.5-flash-preview-04-17",
                history=google_messages,
                config=config
            )   # context caching not available on gemini 2.5
            # catch/retry 500 errors
            retry_limit = 2
            cur_retries = 0
            while cur_retries < retry_limit:
                try:
                    if include_screenshot:
                        response = chat.send_message([types.Part(text=prompt), types.Part.from_bytes(data=screenshot_b64, mime_type="image/png")], config=config)
                    else:
                        response = chat.send_message(prompt, config=config)
                    break
                except ServerError as e:
                    if e.code != 500:
                        raise e
                    cur_retries += 1
            summary_text = " ".join(
                x.text for x in response.candidates[0].content.parts if x.text is not None
            )
        elif model == "OPENAI":
            if include_history:
                messages = copy.deepcopy(self.openai_message_history)
            if include_screenshot:
                messages += [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{screenshot_b64}",
                            }
                        ],
                    },
                ]
            else:
                messages += [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            }
                        ],
                    }
                ]
            response = self.openai_client.responses.create(
                model="o3",
                input=messages,
                instructions=instructions,
                max_output_tokens=MAX_TOKENS_OPENAI,
                temperature=TEMPERATURE,
                tools=OPENAI_TOOLS
            )
            # Gather Reasoning and tool calls
            response_texts = ""
            for chunk in response.output:
                if isinstance(chunk, responses.ResponseOutputMessage):
                    try:
                        response_texts += "\n".join(x.text for x in chunk.content)
                    except AttributeError:
                        # This was probably refused for safety reasons. Wait what?
                        breakpoint()
            summary_text = response_texts
        return summary_text
    
    def agentic_summary(self):
        self.text_display.add_message(f"[Agent] Generating Facts Analysis, standby...")

        memory_info, location, coords = self.emulator.get_state_from_memory()
        try:
            previous_summary = self.message_history[0]["content"][0]["text"]
        except TypeError:
            previous_summary = "Start of the Game!"
        last_checkpoints = '\n'.join(self.checkpoints[-10:])
        prompt = f"""
Here is key game information:

RAM Information: {memory_info}

ASCII MAP: {self.update_and_get_full_collision_map(location, coords)}

Last 10 Checkpoints: {last_checkpoints}

Previous game summary: {previous_summary}

A game screenshot is attached.
"""

        # Get the FACTS
        response1 = self.prompt_text_reply(META_KNOWLEDGE_PROMPT, prompt, True, MODEL, True)
        logger.info(f"Facts Stage 1: {response1}")
        # Clean Facts
        response2 = self.prompt_text_reply(META_KNOWLEDGE_CLEANUP_PROMPT, response1, False, MODEL, False)
        logger.info(f"Facts Stage 2: {response2}")
        # Summarize for real
        response3 = self.prompt_text_reply(META_KNOWLEDGE_SUMMARIZER, response2, True, MODEL, False)
        self.text_display.add_message(f"Final Summary: {response3}")
        with open("agentic_summary.txt", "w", encoding="utf-8") as fw:
            fw.write(response1 + "\n\n" + response2 + "\n\n" + response3)

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
                        "text": f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {response3}"
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
                        "text": (f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {response3}" +
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
        
    def stop(self):
        """Stop the agent."""
        self.running = False
        self.emulator.stop()


if __name__ == "__main__":
    # Get the ROM path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path = os.path.join(os.path.dirname(current_dir), "pokemon.gb")

    # Create and run agent
    agent = SimpleAgent(rom_path)

    try:
        steps_completed = agent.run(num_steps=10)
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    finally:
        agent.stop()