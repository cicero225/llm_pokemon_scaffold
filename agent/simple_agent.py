import base64
import copy
import io
import json
import logging
import numpy as np
import os
import pickle
from PIL import ImageDraw, ImageFilter, Image
import threading
import warnings

from config import *
from agent.prompts import *
from secret_api_keys import *
from agent.emulator import Emulator
from agent.small_task_agent import SmallTaskAgent
from agent.tool_definitions import *
from agent.utils import convert_anthropic_message_history_to_google_format, extract_tool_calls_from_gemini

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

FRIENDLY_MODEL_NAME_LOOKUP = {
    "CLAUDE": "Claude",
    "GEMINI": "Gemini",
    "OPENAI": "Openai's o3"
}


BASE_IMAGE_SIZE = (160, 144)  # In tiles, 16 x 10 columns and 16 x 9 rows

# Handles making an automatically updating collision map of an area as the model paths through it.
# This structure assumes there's only one type of entry per tile, but actually tiles can be both IMPASSABLE and SPRITES or a Player or Sprite can be standing on a warp.
# This is an oversight that should maybe be fixed, but is hard to fix without a lot of restructuring.
class LocationCollisionMap:
    def __init__(self, initial_collision_map: np.ndarray, initial_sprite_locations: set[tuple[tuple[int, int], int]], initial_coords: tuple[int, int], initial_tile_map: np.ndarray, nearby_warps: Optional[list[tuple[int, int]]]=None):
        # initial_collision_map is a 9 x 10 player-centered collision map which is 0 is impassable and 1 otherwise
        # Internally we store an expanding map based on locations we've been, with -1 in unknown spots, 2 in sprite locations, 3 in player location, 4 is item locations, 5 is warps, and otherwise 0/1 as well.
        # We just accept that moving NPC locations are going to be inaccurate.
        # Note that while player coords are column, row, by default what we get from the collision map tooling is row, column
        sprite_lookup = {x: y for x, y in initial_sprite_locations}
        
        self.player_coords = initial_coords
        self.col_offset = initial_coords[0] - 4
        self.row_offset = initial_coords[1] - 4
        # tile_map is a game_area() from the emulator that contains tile information. Used to extract things like grass or ledges.
        # tile_map is 18 x 20, to allow for the possibiltiy of multiple thinsg in the same place. Confusingly, it is row first.
        # Grass is 338.
        self.internal_map = -np.ones((10, 9), dtype=np.int8) # We make our map the "proper" order that everything else is.
        self.tile_map = np.zeros((10, 9), dtype=np.int8)   # for now 1 is grass, 0 is anything else.
        for row in range(9):
            for col in range(10):
                if np.any(initial_tile_map[row*2:row*2+2, col*2:col*2+2] == 338):  # grass
                    self.tile_map[col][row] = 1
                if row == 4 and col == 4:  # player character
                    self.internal_map[col][row] = 3
                elif nearby_warps is not None and (col + self.col_offset, row + self.row_offset) in nearby_warps:
                    self.internal_map[col][row ] = 5
                    continue
                elif (col, row) in sprite_lookup:
                    if sprite_lookup[(col, row)] in [122, 123]:  # This is pokeball
                        self.internal_map[col][row] = 4
                    else:
                        self.internal_map[col][row] = 2
                else:
                    self.internal_map[col][row] = initial_collision_map[row][col]
        self.distances: dict[tuple[int, int], int] = {}
        self.adjacent_to_unexplored_or_edge: set[tuple[int, int]] = set()
        self.discovered_edges: set[str] = set()

    def mark_discovered_edge(self, direction: Literal["up", "down", "left", "right"]):  # some maps have transitions from other maps. In that situation we need to mark everything beyond the boundary in the direction as explored.
        self.discovered_edges.add(direction)

    def update_map(self, collision_map: np.ndarray, sprite_locations: set[tuple[tuple[int, int], int]], coords: tuple[int, int], tile_map: np.ndarray, nearby_warps: Optional[list[tuple[int, int]]]=None):
        # Remove the previous player marker. Most convenient to do it right now.
        self.internal_map[self.player_coords[0] - self.col_offset][self.player_coords[1] - self.row_offset] = 1
        self.player_coords = coords

        sprite_lookup = {x: y for x, y in sprite_locations}
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
        self.tile_map = np.pad(self.tile_map, pad_width=[(expand_col_front, expand_col_back), (expand_row_front, expand_row_back)], constant_values=0)

        self.col_offset = min(new_min_col, self.col_offset)
        self.row_offset = min(new_min_row, self.row_offset)

        # Now update the map
        local_col_offset = new_min_col - self.col_offset
        local_row_offset = new_min_row - self.row_offset
        for row in range(9):
            for col in range(10):
                if np.any(tile_map[row*2:row*2+2, col*2:col*2+2] == 338):  # grass
                    self.tile_map[col][row] = 1
                if row == 4 and col == 4:  # player character. It's okay to wipe warp information temporarily because it's almost certainly coming back.
                    self.internal_map[col + local_col_offset][row + local_row_offset] = 3
                    continue
                if self.internal_map[col + local_col_offset][row + local_row_offset] == 5:  # otherwise we don't wipe this for any reason
                    continue
                # Warp takes precendence over anything else below.
                if nearby_warps is not None and (col + new_min_col, row + new_min_row) in nearby_warps:
                    self.internal_map[col + local_col_offset][row + local_row_offset] = 5
                    continue
                # if self.internal_map[col + local_col_offset][row + local_row_offset] != -1:
                    # continue  # No need to set, and if we do it's just going to lead to extra sprites for moving NPCs
                if (col, row) in sprite_lookup:
                    if sprite_lookup[(col, row)] in [122, 123]:
                        self.internal_map[col + local_col_offset][row + local_row_offset] = 4
                    else:
                        self.internal_map[col + local_col_offset][row + local_row_offset] = 2
                else:
                    self.internal_map[col + local_col_offset][row + local_row_offset] = collision_map[row][col]
        self.update_distances()

    def update_distances(self):
        self.distances, self.adjacent_to_unexplored_or_edge = self.compute_effective_distance_to_tiles()

    def compute_effective_distance_to_tiles(self) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]]:
        # Basically do a distance fill
        self.discovered_edges = set(["down"])
        depth = 200
        visited_tiles = set([self.player_coords])
        cur_tiles = set([self.player_coords])
        distances: dict[tuple[int, int], int] = {}
        adjacent_to_unexplored_or_edge: set[tuple[int, int]] = set()
        for d in range(depth):
            new_tiles: set[tuple[int, int]] = set()
            for tile in cur_tiles:
                # if the tile itsellf is "off the edge" from a explored area then it's not an exploration tile.
                valid_exploration_tile = not ((tile[0] - self.col_offset - 4 < min(0, self.col_offset) and "left" in self.discovered_edges) or (tile[1] - self.row_offset - 4 < min(0, self.row_offset) and "up" in self.discovered_edges) or (tile[0] - self.col_offset + 5 > self.internal_map.shape[0] - 1 and "right" in self.discovered_edges) or (tile[1] - self.row_offset + 4 > self.internal_map.shape[1] - 1 and "down" in self.discovered_edges))         
                candidate_tiles = ((tile[0] + 1, tile[1]), (tile[0] - 1, tile[1]), (tile[0], tile[1] + 1), (tile[0], tile[1] - 1))  # I feel like there's a smarter way
                for candidate in candidate_tiles:
                    shifted_col = candidate[0] - self.col_offset
                    shifted_row = candidate[1] - self.row_offset
                    if (shifted_col < min(0, self.col_offset) and "left" not in self.discovered_edges) or (shifted_row < min(0, self.row_offset) and "up" not in self.discovered_edges) or (shifted_col > self.internal_map.shape[0] - 1 and "right" not in self.discovered_edges) or (shifted_row > self.internal_map.shape[1] - 1 and "down" not in self.discovered_edges):
                        if valid_exploration_tile:
                            adjacent_to_unexplored_or_edge.add(tile)
                    if shifted_col < 0 or shifted_row < 0 or shifted_col > self.internal_map.shape[0] - 1 or shifted_row > self.internal_map.shape[1] - 1:
                        continue  # candidate is out of bounds
                    if self.internal_map[shifted_col][shifted_row] == -1:
                        if valid_exploration_tile:
                            adjacent_to_unexplored_or_edge.add(tile)
                    if candidate in visited_tiles:
                        continue
                    if self.internal_map[shifted_col][shifted_row] in (1, 5):   # the only passable scenarios
                        new_tiles.add(candidate)
                        distances[candidate] = d + 1
                    visited_tiles.add(candidate)
            cur_tiles = new_tiles
        return distances, adjacent_to_unexplored_or_edge

    def generate_buttons_to_coord(self, col: int, row: int) -> Optional[list[str]]:
        starting_distance = self.distances.get((col, row))
        if starting_distance is None:
            return None # invalid
        distance = starting_distance
        button_list = []
        # Basically look for tiles that are labelled with successively lower numbers
        while distance > 0:
            # just pick whichever happens to work first.
            left = self.distances.get((col - 1, row))
            if (left and left == distance - 1) or (col - 1, row) == self.player_coords:
                button_list.append("right")
                col -= 1
                distance -= 1
                continue
            right = self.distances.get((col + 1, row))
            if (right and right == distance - 1) or (col + 1, row) == self.player_coords:
                button_list.append("left")
                distance -= 1
                col += 1
                continue
            up = self.distances.get((col, row - 1))
            if (up and up == distance - 1) or (col, row - 1) == self.player_coords:
                button_list.append("down")
                distance -= 1
                row -= 1
                continue
            down = self.distances.get((col, row + 1))
            if (down and down == distance - 1) or (col, row + 1) == self.player_coords:
                button_list.append("up")
                distance -= 1
                row += 1
                continue
            breakpoint()
        
        # now reverse and return
        button_list.reverse()
        return button_list
    
    @staticmethod
    def make_ascii_segment(input_str: str, width: int, col: int, row: int):
        # Basically output ascii map blocks of a consistent width, using a given input_str and local coordinates. Also adds | on the front side and includes it in the width.
        base_str = f"{input_str}({col},{row})"
        # pads always at the end.
        if len(base_str) > width - 1:
            warnings.warn("Not enough space to fit this!")
        base_str += (width - 1 - len(base_str))*" "
        return f"|{base_str[:width]}"

    def to_ascii(self, local_location_tracker: Optional[list[list[bool]]]=None, direction: Optional[str]=None, local_label_archive: Optional[dict[int, dict[int, str]]]=None) -> str:
        # We prepare two identical versions simultaneously: A readable nice ASCII for humans, and the long-winded one for models

        horizontal_labels = list(range(self.col_offset, self.col_offset+self.internal_map.shape[0]))

        # TODO: Make the row_width adaptive.
        row_width = 120
        horizontal_border = "       +" + "".join("Column " + str(x) + " "*(row_width - len(str(x)) - 7) for x in horizontal_labels) + "+"
        horizontal_border_human = "       +" + "".join(str(x) + " "*(4-len(str(x))) for x in horizontal_labels) + "+"

        lines = []
        lines_human = []
        # Add legend to human version
        if local_location_tracker:
            lines_human.extend(
                [
                    "",
                    "Legend:",
                    "██ - Wall/Obstacle",
                    "·· - Path/Walkable",
                    "SS - Sprite",
                    "PP - Player Character",
                    "II - Item"
                    "xx - AVOID GOING HERE - Already Explored",
                    "uu - Blank = Unknown/Unvisited",
                    "ww - Warp",
                    "Numbers - How many tiles away this tile is to reach."
                    "cc - CHECK HERE - next to Unexplored.",
                    ":: - Unreachable?"
                ]
            )
        else:
            lines_human.extend(
                [
                    "",
                    "Legend:",
                    "██ - Wall/Obstacle",
                    "·· - Path/Walkable",
                    "SS - Sprite",
                    "PP - Player Character",
                    "II - Item",
                    "xx - AVOID GOING HERE - Already Explored",
                    "uu - Blank = Unknown/Unvisited",
                    "cc - CHECK HERE - next to Unexplored."
                ]
            )

        lines += [f"({self.col_offset}, {self.row_offset})", horizontal_border]
        lines_human += [f"({self.col_offset}, {self.row_offset})", horizontal_border_human]
        for row_num, this_row in enumerate(self.internal_map.transpose()):  # transposing makes printing easier
            real_row = self.row_offset + row_num
            row = f"Row: {str(real_row) + ' ' * (2 - len(str(real_row)))}"
            row_human = row + "|"
            for col_num, col in enumerate(this_row):
                real_col = self.col_offset + col_num

                row_piece = ""
                if self.tile_map[col_num][row_num] == 1 and col > 0:
                    row_piece = "Grass; "
                else:
                    row_piece = "Not grass; "
                if local_label_archive is not None:
                    this_label_row = local_label_archive.get(real_row)
                    if this_label_row is not None:
                        maybe_label = this_label_row.get(real_col)
                        if maybe_label is not None:
                            row_piece += "Label: " + maybe_label + "; "
                # Here we face the price of previous mistakes. We wouldn't need a nearby_warps argument if it weren't
                # for the case that we can't log both a player character and warp or sprite in the same spot.
                # Because of this, we now have an edge case if the player character is on a warp, or a sprite is.
                if col == -1:
                    row += self.make_ascii_segment(row_piece + "Unexplored", row_width, real_col, real_row)
                    row_human += " uu "
                elif col == 0:
                    row += self.make_ascii_segment(row_piece + "Impassable", row_width, real_col, real_row)
                    row_human += " ██ "
                elif col == 1: 
                    # Potentially place a distance marker:
                    row_piece_human = ""
                    distance = self.distances.get((real_col, real_row))
                    if distance:  # removes 0 and None
                        row_piece += "StepsToReachFromPlayer:" + str(distance) + " " * (4 - len(str(distance))) + " "
                        row_piece_human += str(distance) + " " * (4 - len(str(distance)))
                    elif distance is None:
                        row_piece += "Currently Seems Unreachable; "
                        row_piece_human += " :: "
                    if local_location_tracker and real_col > -1 and real_row > -1 and real_col < len(local_location_tracker) and real_row < len(local_location_tracker[real_col]) and local_location_tracker[real_col][real_row]:
                        row_piece += "Explored"
                        if not row_piece_human:
                            row_human += " xx "
                    else:
                        row_piece += "Passable"
                        if (real_col, real_row) in self.adjacent_to_unexplored_or_edge:
                            row_piece += " CHECK HERE TO EXPLORE "
                            row_piece_human = " cc "
                        if not row_piece_human:
                            row_human += " ·· "
                    row += self.make_ascii_segment(row_piece, row_width, real_col, real_row)
                    row_human += row_piece_human
                elif col == 2:
                    row += self.make_ascii_segment(row_piece + "NPC/Object", row_width, real_col, real_row)
                    row_human += " SS "
                elif col == 3:
                    row += self.make_ascii_segment(row_piece + f"PLAYER (Facing {direction})" , row_width, real_col, real_row)
                    row_human += " PP "
                elif col == 4:
                    distance = self.distances.get((real_col, real_row))
                    if distance is None:
                        row_piece += "Currently Seems Unreachable; "
                    row += self.make_ascii_segment(row_piece + f"Item" , row_width, real_col, real_row)
                    row_human += " II "
                elif col == 5:
                    distance = self.distances.get((real_col, real_row))
                    if distance is None:
                        row_piece += "Currently Seems Unreachable; "
                    row += self.make_ascii_segment(row_piece + "Warp", row_width, real_col, real_row)
                    row_human += " ww "
            row += f"|{str(real_row)}"
            row_human += f"|{str(real_row)}"
            lines.append(row)
            lines_human.append(row_human)
        lines.append(horizontal_border + f"({self.col_offset + self.internal_map.shape[0] - 1}, {self.row_offset + self.internal_map.shape[1] - 1})")
        lines_human.append(horizontal_border_human + f"({self.col_offset + self.internal_map.shape[0] - 1}, {self.row_offset + self.internal_map.shape[1] - 1})")


        # Join all lines with newlines
        output = "\n".join(lines)
        with open("mapping_log.txt", "w", encoding="utf-8") as fw:
            fw.write("\n".join(lines_human))
            fw.write("\n\n" + "MODEL VERSION:" +"\n\n")
            fw.write(output)
        return output
    
    def is_in_map(self, col: int, row: int) -> bool:
        if col < self.col_offset or row < self.row_offset or col - self.col_offset > self.internal_map.shape[0] - 1 or row - self.row_offset > self.internal_map.shape[1] - 1:
            return False
        if self.internal_map[col - self.col_offset][row - self.row_offset] == -1:
            return False
        return True

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
    def __init__(
        self, 
        rom_path, 
        headless=True, 
        sound=False, 
        max_history=60, 
        load_state=None, 
        location_history_length=40, 
        location_archive_file_name: Optional[str]=None, 
        use_full_collision_map: bool=True,
        pyboy_main_thread: bool=False
    ):
        """Initialize the simple agent.

        Args:
            rom_path: Path to the ROM file
            headless: Whether to run without display
            sound: Whether to enable sound
            max_history: Maximum number of messages in history before summarization
        """
        self.emulator = Emulator()
        self.pyboy_main_thread = pyboy_main_thread
        self.emulator_init_kwargs = {"rom_path": rom_path, "headless": headless, "sound": sound, "pyboy_main_thread": self.pyboy_main_thread}
        if not self.pyboy_main_thread:
            self.emulator.initialize(**self.emulator_init_kwargs)
        if MODEL == "CLAUDE" or MINI_MODEL == "CLAUDE":
            self.anthropic_client = Anthropic(api_key=API_CLAUDE, max_retries=10)
        if MODEL == "GEMINI" or MINI_MODEL == "GEMINI":
            self.gemini_client = genai.Client(api_key=API_GOOGLE)
        if MODEL == "OPENAI" or MINI_MODEL == "OPENAI":
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
        self.checkpoints = []  # A long-running list of achievements, used to track internal progress.
        self.detailed_navigator_mode = False
        self.navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}]
        self.openai_navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}]
        self.steps_since_location_shift = 0
        self.no_navigate_here = ""
        self.navigation_location = ""
        self._steps_completed = 0
        self.load_state = load_state
        self.sub_agent: Optional[SmallTaskAgent] = None
        self.button_history = []
        self.navigator_goal: str = ""
        self.exploration_repeats = 0
        self.is_blackout = False

        if load_state and not self.pyboy_main_thread:
            logger.info(f"Loading saved state from {load_state}")
            self.emulator.load_state(load_state)
            self.load_location_archive(location_archive_file_name)
        elif load_state:
            self.load_location_archive(location_archive_file_name)


    # This does the overlay for the model.
    def get_screenshot_base64(
            self, screenshot: Image.Image, upscale=1, add_coords: bool=True,
            player_coords: Optional[tuple[int, int]]=None, location: Optional[str]=None, relative_square_size=8, show_nearby_warps: bool=True):
        """Convert PIL image to base64 string."""
        # Resize if needed
        if upscale > 1:
            new_size = (screenshot.width * upscale, screenshot.height * upscale)
            screenshot = screenshot.resize(new_size)

        past_locations = self.location_history
        location_labels = self.label_archive.get(location)
        if location is not None:
            # this sucks man
            for key, value in self.label_archive.items():
                if location.lower() == key.lower():
                    location_labels = ({} if location_labels is None else location_labels) | value
                    break
        if location_labels is None:
            location_labels = {}
        local_location_tracker = self.location_tracker.get(location, [])

        collision_map = self.emulator.pyboy.game_wrapper.game_area_collision()
        downsampled_terrain = self.emulator._downsample_array(collision_map)

        sprite_locations_and_ids = {x: y for x, y in self.emulator.get_sprites()}

        if not self.emulator.get_in_combat():
            shape = screenshot.size
            # Draw some eye-searing lines across the image that nonetheless might make it more obvious to the LLM that this is a grid.
            for x in range(0, shape[0], shape[0]//10):
                ImageDraw.Draw(screenshot).line(((x, 0), (x, shape[1] - 1)), fill=(255, 0, 0))
            for y in range(0, shape[1], shape[1]//9):
                ImageDraw.Draw(screenshot).line(((0, y), (shape[0] - 1, y)), fill=(255, 0, 0))

            # add coordinate labels (note: if scale is too small it may be unreadable)
            # The assumption is the central square is the player's current location, which is 4, 4
            # Rows 0 - 8, Cols 0 - 9
            if add_coords:
                assert player_coords is not None
                if show_nearby_warps:
                    all_warps = self.emulator.get_warps()
                    nearby_warps = []
                    for entry in all_warps:
                        if (entry[0] - player_coords[0] < 6 or player_coords[0] - entry[0] < 5) and abs(entry[1] - player_coords[1]) < 5:
                            nearby_warps.append(entry)
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
                        try:
                            label, _ = label.split(": [LOGGED BY TOOL] ")  # Dialogue labels we cut off the second part since it's way too big for an image.
                        except Exception as e:
                            pass
                        tile_label = f"{str(real_col)}, {str(real_row)}"
                        if label:
                            tile_label += "\n" + label
                        if show_nearby_warps and (real_col, real_row) in nearby_warps:
                            tile_label += "\n" + "WARP"
                        if (col, row) not in sprite_locations_and_ids:
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
                            if sprite_locations_and_ids[(col, row)] in [122, 123]:  # this is pokeball
                                tile_label += "\n" + "ITEM"
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
            pickle.dump(self.detailed_navigator_mode, fw)
            pickle.dump(self.navigator_message_history, fw)
            pickle.dump(self.openai_navigator_message_history, fw)
            pickle.dump(self.steps_since_location_shift, fw)
            pickle.dump(self.no_navigate_here, fw)
            pickle.dump(self.navigation_location, fw)
            # We strip the internal reference temporarily, or else pickle will fail
            if self.sub_agent is not None:
                self.sub_agent.senior_agent = None
                pickle.dump(self.sub_agent, fw)
                self.sub_agent.senior_agent = self
            else:
                pickle.dump(self.sub_agent, fw)  # We still need to dump it for consistency in variable list.
            pickle.dump(self.button_history, fw)
            pickle.dump(self.navigator_goal, fw)
            pickle.dump(self.exploration_repeats, fw)
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
                    self.detailed_navigator_mode = pickle.load(fr)
                    self.navigator_message_history = pickle.load(fr)
                    self.openai_navigator_message_history = pickle.load(fr)
                    self.steps_since_location_shift = pickle.load(fr)
                    self.no_navigate_here = pickle.load(fr)
                    self.navigation_location = pickle.load(fr)
                    self.sub_agent = pickle.load(fr)
                    if self.sub_agent is not None:
                        self.sub_agent.senior_agent = self
                    self.button_history = pickle.load(fr)
                    self.navigator_goal = pickle.load(fr)
                    self.exploration_repeats = pickle.load(fr)
                    for x in self.full_collision_map.values():
                        try:
                            x.tile_map
                        except AttributeError:
                            x.tile_map = np.zeros(shape=x.internal_map.shape, dtype=np.int8)
                except Exception:
                    pass
            breakpoint()
        except FileNotFoundError:
            logger.warn("No Location archive! Making new one...")
        if not isinstance(self.message_history[-1]['content'], str):
            for entry in self.message_history[-1]['content']:  # In case we somehow got interrupted mid tool use for anything that isn't a subagent.
                if entry['type'] == 'tool_use' and entry["name"] != "use_subagent":
                    self.message_history.pop()
                    break
        if self.navigator_message_history and 'content' in self.navigator_message_history[-1] and not isinstance(self.navigator_message_history[-1]['content'], str):
            for entry in self.navigator_message_history[-1]['content']:
                if entry['type'] == 'tool_use':
                    self.navigator_message_history.pop()
                    break
        if self.sub_agent is not None:
            if not isinstance(self.sub_agent.history[-1]['content'], str):
                for entry in self.sub_agent.history[-1]['content']:
                    if entry['type'] == 'tool_use':
                        self.sub_agent.history.pop()
                        break
        # Edge case: if switching from gemini to Claude, we may have to fix a few broken tool_calls...
        unique_str = 0
        for message in self.message_history:
            content = message.get("content")
            if isinstance(content, str):
                continue
            if content:
                for entry in content:
                    if entry["type"] == "tool_use" and entry["id"] is None:
                        entry["id"] = str(unique_str)
                    if "tool_use_id" in entry and entry["tool_use_id"] is None:
                        entry["tool_use_id"] = str(unique_str)
                        unique_str += 1
        unique_str = 0
        for message in self.navigator_message_history:
            content = message.get("content")
            if isinstance(content, str):
                continue
            if content:
                for entry in content:
                    if entry["type"] == "tool_use" and entry["id"] is None:
                        entry["id"] = str(unique_str)
                    if "tool_use_id" in entry and entry["tool_use_id"] is None:
                        entry["tool_use_id"] = str(unique_str)
                        unique_str += 1
        # TODO: Code for being able to restart gemini/openai with an interrupted tool call. Currently may glitch out in that scenario.

    # Save tokens...
    @staticmethod
    def strip_text_map_and_images_from_history(message_history: list[dict[str, Any]], openai_format: bool=False) -> None:
        # We go through everything that's not the most recent tool_result/message and clip out images and
        # text_based maps to save tokens.
        image_type_str = "input_image" if openai_format else "image"
        text_type_str = "input_text" if openai_format else "text"
        for message in message_history[:-2]:
            if openai_format and not isinstance(message, dict) :
                continue
            # So obnoxiously, openai handles function outputs totally differently
            if openai_format and message.get('type') == "function_call_output":
                maybe_content = json.loads(message['output'])
            else:
                maybe_content = message["content"]
                if isinstance(maybe_content, str):
                    continue
            if maybe_content is not None:
                new_content = []
                for entry in maybe_content:
                    if isinstance(entry, str):
                         # we can just assume it's one of a few special messages  we don't have to do anything with.
                        new_content.append(entry)
                        continue
                    if entry["type"] == image_type_str:
                        continue
                    elif entry["type"] == text_type_str:
                        text = entry["text"]
                        try:
                            # Remove everything between [TEXT_MAP] tags
                            first, second = text.split("[TEXT_MAP]")
                            second, third = second.split("[/TEXT_MAP]")
                            entry["text"] = first + "TEXT MAP and sceenshot omitted to save redundancy" + third
                        except Exception:
                            pass
                    elif entry["type"] == "tool_result":
                        sub_content = []
                        for sub_entry in entry["content"]:
                            if sub_entry["type"] == image_type_str:
                                continue
                            elif sub_entry["type"] == text_type_str:
                                text = sub_entry["text"]
                                try:
                                    # Remove everything between [TEXT_MAP] tags
                                    first, second = text.split("[TEXT_MAP]")
                                    second, third = second.split("[/TEXT_MAP]")
                                    sub_entry["text"] = first + "TEXT MAP and sceenshot omitted to save redundancy" + third
                                except Exception:
                                    pass
                            sub_content.append(sub_entry)
                        entry["content"] = sub_content         
                    new_content.append(entry)
                if openai_format and message.get('type') == "function_call_output":
                    message['output'] = json.dumps(new_content)
                else:
                    message["content"] = new_content
            else:
                breakpoint()

    def update_and_get_full_collision_map(self):
        location = self.emulator.get_location()
        coords = self.emulator.get_coordinates()
        collision_map = self.emulator.pyboy.game_wrapper.game_area_collision()
        downsampled_terrain = self.emulator._downsample_array(collision_map)
        local_location_tracker = self.location_tracker.get(location, [])
        all_warps = self.emulator.get_warps()
        nearby_warps = []
        direction = self.emulator.get_direction()
        for entry in all_warps:
            if (entry[0] - coords[0] < 6 or coords[0] - entry[0] < 5) and abs(entry[1] - coords[1]) < 5:
                nearby_warps.append(entry)

        this_location = self.label_archive.get(location)
        # this sucks man
        for key, value in self.label_archive.items():
            if location.lower() == key.lower():
                this_location = ({} if this_location is None else this_location) | value
                break
        
        # slightly more efficient than setdefault
        this_map = self.full_collision_map.get(location)
        if this_map is None:
            self.full_collision_map[location] = LocationCollisionMap(downsampled_terrain, self.emulator.get_sprites(), coords, self.emulator.pyboy.game_area(), nearby_warps) 
            # Handle edge case: if we're not standing on a warp, then we just came in through a zone transition and
            # need to inform the mapping tool.
            if coords not in all_warps:
                if self.button_history:
                    if self.button_history[-1] == "down":
                        self.full_collision_map[location].mark_discovered_edge("up")
                    elif self.button_history[-1] == "up":
                        self.full_collision_map[location].mark_discovered_edge("down")
                    elif self.button_history[-1] == "left":
                        self.full_collision_map[location].mark_discovered_edge("right")
                    else:
                        self.full_collision_map[location].mark_discovered_edge("left")
            return self.full_collision_map[location].to_ascii(local_location_tracker, direction=direction, local_label_archive=this_location)
        else:
            this_map.update_map(downsampled_terrain, self.emulator.get_sprites(), coords, self.emulator.pyboy.game_area(), nearby_warps)
            return this_map.to_ascii(local_location_tracker, direction=direction, local_label_archive=this_location)
        
    def get_all_location_labels(self, location: str) -> list[tuple[tuple[int, int], str]]:
        all_labels: list[tuple[tuple[int, int], str]] = []
        this_location = self.label_archive.get(location)
        # this sucks man
        for key, value in self.label_archive.items():
            if location.lower() == key.lower():
                this_location = ({} if this_location is None else this_location) | value
                break
        if this_location is not None and this_location:
            max_row = max(this_location.keys())
            for nearby_row in range(max_row + 1):
                this_row = this_location.get(nearby_row)
                if this_row is not None:
                    max_col = max(this_row.keys())
                    for nearby_col in range(max_col + 1):
                        this_col = this_row.get(nearby_col)
                        if this_col is not None:
                            all_labels.append(((nearby_col, nearby_row), this_col))  # Note that we only care about our current location
        return all_labels
    
    def press_buttons(self, buttons: list[str], wait: bool, tool_id: str, include_text_map: bool=True, is_subtool: bool=False, suppress_button_repetition_warning: bool=False, interrupt_if_combat_or_dialogue: bool=False) -> dict[str, Any]:
        self.text_display.add_message(f"[Buttons] Pressing: {buttons} (wait={wait})")
        

        button_repetition_warning = ""
        if not is_subtool:
            self.button_history.extend(buttons)
            if len(self.button_history) > MAX_BUTTON_HISTORY:
                self.button_history = self.button_history[MAX_BUTTON_HISTORY - len(self.button_history):]
            if not suppress_button_repetition_warning:   
                a_count = 0
                for button in reversed(self.button_history):
                    if button != "a":
                        break
                    a_count += 1
                    if a_count > MAX_A_PRESS:
                        button_repetition_warning = f"\nWARNING: You have pressed A at least {MAX_A_PRESS} times in a row. This is a task that may be suitable for a subagent."
                dir_count = 0
                for button in reversed(self.button_history):
                    if button not in ["left", "right", "down", "up"]:
                        break
                    dir_count += 1
                    if dir_count > MAX_DIRECTION_PRESS:
                        button_repetition_warning = f"\nWARNING: You have pressed a direction button at least {MAX_A_PRESS} times in a row. It looks like you may be navigating or trying to talk. Use navigate_to_coordinate or talk_to_npc_or_pick_up_item instead."

        # This really sucks, but going one by one lets us catch location changes better,
        last_location = None
        for button in buttons:
            result, last_coords = self.emulator.press_buttons([button], wait, wait_for_finish=True)  # we need to wait for this to finish or the map may try to update mid move, which is bad.
            location = self.emulator.get_location()
            if location != last_location:
                self.exploration_repeats = 0
                coords = self.emulator.get_coordinates()
                all_warps = self.emulator.get_warps()
                if location not in self.full_collision_map:
                    # We do this as close to when the location transition happens as possible or else the discovered edge will not be correct.
                    nearby_warps = []
                    for entry in all_warps:
                        if (entry[0] - coords[0] < 6 or coords[0] - entry[0] < 5) and abs(entry[1] - coords[1]) < 5:
                            nearby_warps.append(entry)
                    collision_map = self.emulator.pyboy.game_wrapper.game_area_collision()
                    downsampled_terrain = self.emulator._downsample_array(collision_map)
                    self.full_collision_map[location] = LocationCollisionMap(downsampled_terrain, self.emulator.get_sprites(), coords, self.emulator.pyboy.game_area(), nearby_warps) 
                # Handle edge case: if we're not standing on a warp, then we just came in through a zone transition and
                # need to inform the mapping tool.
                if coords not in all_warps:
                    if self.button_history:
                        if self.button_history[-1] == "down":
                            self.full_collision_map[location].mark_discovered_edge("up")
                        elif self.button_history[-1] == "up":
                            self.full_collision_map[location].mark_discovered_edge("down")
                        elif self.button_history[-1] == "left":
                            self.full_collision_map[location].mark_discovered_edge("right")
                        else:
                            self.full_collision_map[location].mark_discovered_edge("left")
                last_location = location
            # Brutish hack but it works. We need this to prevent weird labels being set thanks to
            # blackout teleportation.
            # Note a nuance here: this is not allowed to set it False, only to True. False can only be set in the main loop that also does labeling.
            if self.emulator.get_active_dialog() and "blacked out" in self.emulator.get_active_dialog().lower():
                self.is_blackout = True
            if interrupt_if_combat_or_dialogue:
                if self.emulator.get_active_dialog() or self.emulator.get_in_combat():
                    break

        self.last_coords = last_coords

        location = self.emulator.get_location()
        
        # Get game state from memory after the action
        memory_info, location, coords = self.emulator.get_state_from_memory(display_warp_functor=self.full_collision_map[location].is_in_map if location in self.full_collision_map else None)
        # Log the memory state after the tool call
        logger.info(f"[Memory State after action]")
        logger.info(memory_info)
        
        if include_text_map:
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

        all_labels = self.get_all_location_labels(location)


        # Return tool result as a dictionary
        # OPENAI doesn't know what to do with too much information here.
        if MODEL == "OPENAI":
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": (
                        f"Pressed buttons: {', '.join(buttons)}" + button_repetition_warning
                    )}
                ],
            }
        else:
            # This is probably worth a refactor
            if is_subtool:
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
                        {"type": "text", "text": f"\nLabeled nearby locations: {','.join(f'{label_coords}: {label}' for label_coords, label in all_labels)}"}
                    ]
                content.append({"type": "text", "text": f"\nHere are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}"})
                if not self.emulator.get_in_combat() and include_text_map:
                    content.append({"type": "text", "text": "Here is a map of this RAM location compiled so far:\n\n[TEXT_MAP]" + self.update_and_get_full_collision_map() + "\n\n[/TEXT_MAP]"})
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                }
            # Get a fresh screenshot after executing the buttons
            if self.use_navigator_this_round:
                # In navigator mode it gets confused if the screenshot/text_based isn't in the user prompt, so we trim it to save tokens.
                # TODO: That may not actually be true; there was another coding error. But this is already done so...
                last_checkpoints = '\n'.join(self.checkpoints[-10:])
                content = [
                        {"type": "text", "text": f"Navigation result: {result}"},
                        {"type": "text", "text": button_repetition_warning},
                        {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                        {"type": "text", "text": f"\nLabeled nearby locations: {','.join(f'{coords}: {label}' for coords, label in all_labels)}"},
                        {"type": "text", "text": f"Here are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}"},
                        {"type": "text", "text": f"Here are your last 10 checkpoints:\n{last_checkpoints}"},
                        {"type": "text", "text": f"You have been in this location for {self.steps_since_location_shift} steps"}
                    ]
                if location in self.full_collision_map:
                    content.append({"type": "text", "text": f"\nCHECK HERE TO EXPLORE {','.join(f'({x[0]}, {x[1]})' for x in self.full_collision_map[location].adjacent_to_unexplored_or_edge)}"})
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                }
            else:
                screenshot = self.emulator.get_screenshot()
                screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                last_checkpoints = '\n'.join(self.checkpoints[-10:])
                content = [
                        {"type": "text", "text": f"Pressed buttons: {', '.join(buttons)}"},
                        {"type": "text", "text": button_repetition_warning},
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
                        {"type": "text", "text": f"Here are your last 10 checkpoints:\n{last_checkpoints}"},
                        {"type": "text", "text": f"You have been in this location for {self.steps_since_location_shift} steps"}
                    ]
                if location in self.full_collision_map:
                    content.append({"type": "text", "text": f"\nCHECK HERE TO EXPLORE {','.join(f'({x[0]}, {x[1]})' for x in self.full_collision_map[location].adjacent_to_unexplored_or_edge)}"})
                if self.emulator.get_in_combat():  # Only possible if navigator mode has been running.
                    content.append({"type": "text", "text": "NOTE: A Navigator version of Claude has been handling overworld movement for you, so your location may have shifted substantially. Please handle this battle for now."})
                if not self.emulator.get_in_combat() and self.use_full_collision_map and include_text_map:
                    content.append({"type": "text", "text": "Here is a map of this RAM location compiled so far:\n\n[TEXT_MAP]" + self.update_and_get_full_collision_map() + "\n\n[/TEXT_MAP]"})
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content,
                }


    # TODO: An obvious refactor would be to move some of these into their own functions.
    def process_tool_call(self, tool_call, model: str):
        """Process a single tool call."""
        tool_name = tool_call.name
        if model == "CLAUDE":
            tool_input = tool_call.input
            tool_id = tool_call.id
        elif model == "GEMINI":
            tool_input = tool_call.args
            tool_id = tool_call.id
        elif model == "OPENAI":
            tool_input = json.loads(tool_call.arguments)
            tool_id = tool_call.call_id
            self.text_display.add_message(f"[Text] {tool_input['explanation_of_action']}")
        logger.info(f"Processing tool call: {tool_name}")

        if tool_name == "press_buttons":
            buttons = tool_input["buttons"]
            wait = tool_input.get("wait", True)
            return self.press_buttons(buttons, wait, tool_id)
        elif tool_name == "navigate_to":  # Unused for now
            row = tool_input["row"]
            col = tool_input["col"]
            location = self.emulator.get_location()
            memory_info, location, coords = self.emulator.get_state_from_memory(display_warp_functor=self.full_collision_map[location].is_in_map if location in self.full_collision_map else None)
            self.text_display.add_message(f"[Navigation] Navigating to: ({col}, {row})")  # 8, 3 -> 6, 4 is 2, 5
            
            # The navigator goes to location on screen, with 0,0 at the top left.
            local_col = col - coords[0] + 4
            local_row = row - coords[1] + 4

            status, path = self.emulator.find_path(local_row, local_col)
            last_coords = coords
            next_coords = coords
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True, wait_for_finish=False)
                    cur_coords = self.emulator.get_coordinates()
                    if cur_coords != next_coords:
                        last_coords = next_coords
                        next_coords = cur_coords
                self.last_coords = last_coords
                result = f"Navigation successful: followed path with {len(path)} steps"
            else:
                result = f"Navigation failed: {status}"
            
            # Get game state from memory after the action
            location = self.emulator.get_location()
            memory_info, location, coords = self.emulator.get_state_from_memory(display_warp_functor=self.full_collision_map[location].is_in_map if location in self.full_collision_map else None)

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

            # TODO: eventually do this more reasonably. For now we do this extraordinarily dumb approach.
            all_labels = self.get_all_location_labels(location)

            # Return tool result as a dictionary
            # OPENAI doesn't know what to do with too much information here.
            if model == "OPENAI":
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
                if self.use_navigator_this_round:
                    # In navigator mode it gets confused if the screenshot/text_based isn't in the user prompt, so we trim it to save tokens.
                    last_checkpoints = '\n'.join(self.checkpoints[-10:])
                    content = [
                            {"type": "text", "text": f"Navigation result: {result}"},
                            {"type": "text", "text": f"\nGame state information from memory after your action:\n{memory_info}"},
                            {"type": "text", "text": f"\nLabeled nearby locations: {','.join(f'{coords}: {label}' for coords, label in all_labels)}"},
                            {"type": "text", "text": f"Here are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}"},
                            {"type": "text", "text": f"Here are your last 10 checkpoints:\n{last_checkpoints}"},
                            {"type": "text", "text": f"You have been in this location for {self.steps_since_location_shift} steps"}
                        ]
                    if location in self.full_collision_map:
                        content.append({"type": "text", "text": f"\nCHECK HERE TO EXPLORE {','.join(f'({x[0]}, {x[1]})' for x in self.full_collision_map[location].adjacent_to_unexplored_or_edge)}"})
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": content,
                    }
                else:
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
                            {"type": "text", "text": f"Here are your last 10 checkpoints:\n{last_checkpoints}"},
                            {"type": "text", "text": f"You have been in this location for {self.steps_since_location_shift} steps"}
                        ]
                    if location in self.full_collision_map:
                        content.append({"type": "text", "text": f"\nCHECK HERE TO EXPLORE {','.join(f'({x[0]}, {x[1]})' for x in self.full_collision_map[location].adjacent_to_unexplored_or_edge)}"})
                    if not self.emulator.get_in_combat() and self.use_full_collision_map:
                        content.append({"type": "text", "text": "Here is an text_based map of this RAM location compiled so far:\n\n" + self.update_and_get_full_collision_map()})
                    if self.emulator.get_in_combat():  # Only possible if navigator mode has been running.
                        content.append({"type": "text", "text": "NOTE: A Navigator version of Claude has been handling overworld movement for you, so your location may have shifted substantially. Please handle this battle for now."})
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": content,
                    }
        elif tool_name == "ask_for_advice":
            topic = tool_input["topic"]
            if topic == "shopping":
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {"type": "text", "text": """
    To find the shopkeeper, look for a NPC that is surrounded completely on all sides by impassable tiles.
    In Pokemon Red, they are usually on the middle-left side of the store, in column 0. You can talk to them directly using "talk_to_npc".
                         
    Systematically search Column 0 in your text map until you find a NPC and use talk_to_npc on that coordinate.
                         """}
    
                    ],
                }
            elif topic == "beating brock":
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {"type": "text", "text": """
    Beating Brock is straightforward with with squirtle or bulbasaur, but considerably more difficult with charmander, especially in Pokemon Red. Some key tips:
    
1. In Pokemon Red, options for other pokemon to help on Brock are limited. There is no Mankey on Route 22 (which is only in Pokemon Yellow) and Nidoran doesn't learn double-kick until Level 43.
    The best option is typically considered evolving a Metapod from Viridian forest into Butterfree for Confusion.
2. Onix and Geodude both have very high defense relative to special defense. Because of this, Ember is better than Scratch, despite being Not Very Effective.
3. Consider training Charmander to Lvl 16 for the evolution to Charmeleon. The significant stat boost can really help.
4. Onix's Bide does damage based on how much take it takes while charging. This is a great time to use growl or other non-damaging moves.
"""}
    
                    ],
                }
            else:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {"type": "text", "text": f"Unknown topic received. No advice to give."}
                    ],
                }
        elif tool_name == "end_navigation":
            reason = tool_input["reason"]
            self.detailed_navigator_mode = False
            reason_str = f"Detailed Navigator has turned itself off. Reason: {reason}"
            self.message_history.append({"role": "user", "content": [{"text": reason_str, "type": "text"}]})  # type: ignore
            self.openai_message_history.append({"role": "user", "content": [{"text": reason_str, "type": "text"}]})
            return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "do_not_append_tool_to_history": True,
                }
        elif tool_name == "explore_direction":
            if self.exploration_repeats > 3:
                return {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [
                            {"type": "text", "text": f"Explore command rejected because of too many consecutive roadblock failures--exploration is almost certainly running into a soft game block. Tool is disabled until your location changes."}
                        ],
                    }
            if self.emulator.get_in_combat():
                return {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [
                            {"type": "text", "text": f"Invalid. Navigation not possible while in combat."}
                        ],
                    }
            dialog = self.emulator.get_active_dialog()
            if dialog is not None:
                return {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [
                            {"type": "text", "text": f"Invalid. Navigation not possible while there is active dialogue on screen."}
                        ],
                    }
            direction = tool_input["direction"]
            num_pokes = 3
            laxness = 3  # How much we will tolerate going in the wrong way.
            # Pick a suggested exploration tile num_pokes, favoring the given direction if possible        
            num_moves = 0
            def get_best_candidate(suggestion_set: set[tuple[int, int]], direction: str):
                location = self.emulator.get_location()
                coords = self.emulator.get_coordinates()
                local_map = self.full_collision_map[location]
                if not local_map.distances:  # this can happen when we cross into a new location, for example.
                    local_map.update_distances()
                if not suggestion_set:
                    # Nothing is unexplored.
                    return None
                if direction == "up":  # we want the lowest row count
                    suggestion = min(suggestion_set, key=lambda x: x[1])
                    if suggestion[1] > coords[1] + laxness:
                        return None
                elif direction == "down": 
                    suggestion = max(suggestion_set, key=lambda x: x[1])
                    if suggestion[1] < coords[1] - laxness:
                        return None
                elif direction == "left":
                    suggestion = min(suggestion_set, key=lambda x: x[0])
                    if suggestion[0] > coords[0] + laxness:
                        return None
                else:
                    suggestion = max(suggestion_set, key=lambda x: x[0])
                    if suggestion[0] < coords[0] - laxness:
                        return None
                return suggestion
            candidates: list[tuple[int, int]] = []
            original_location = self.emulator.get_location()
            result = None
            while num_moves < num_pokes:
                # this should update each time navigate_to_coordinate is called...
                location = self.emulator.get_location()
                if location != original_location:
                    result["content"].append({"type": "text", "text": f"Moved to {' then, '.join(str(x) for x in candidates)}"})
                    result["content"].append({"type": "text", "text": f"Exploration terminated. New location reached."})
                    return result
                suggestion_set = self.full_collision_map[location].adjacent_to_unexplored_or_edge
                if suggestion_set is None:
                    coords = self.emulator.get_coordinates()
                    screenshot = self.emulator.get_screenshot()
                    screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                    return {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": [
                                {"type": "text", "text": f"Aborted. No valid unexplored tiles."}
                                ,
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": screenshot_b64,
                                    },
                                },
                            ],
                        }
                candidate = get_best_candidate(suggestion_set, direction)
                if candidate is None:
                    coords = self.emulator.get_coordinates()
                    screenshot = self.emulator.get_screenshot()
                    screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                    if result is None:
                        return {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": [
                                {"type": "text", "text": f"Aborted. No mapped exploration tiles in the right direction. Try going perpindincular for a bit then try again?"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": screenshot_b64,
                                    },
                                },
                            ],
                        }
                    else:
                        if candidates:
                            result["content"].append({"type": "text", "text": f"Moved to {' then, '.join(str(x) for x in candidates)}"})
                        result["content"].append({"type": "text", "text": f"Aborted. No mapped exploration tiles in the right direction. Try going perpindincular for a bit then try again?"})
                        return result
                candidates.append(candidate)
                # We assume it works, since the map should guarantee that. Even if it doesn't, fail silently for now.
                result = self.navigate_to_coordinate(candidate[0], candidate[1], tool_id, is_subagent=False)
                if self.emulator.get_active_dialog():
                    self.exploration_repeats += 1
                    coords = self.emulator.get_coordinates()
                    screenshot = self.emulator.get_screenshot()
                    screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                    return {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": [
                            {"type": "text", "text": f"Aborted. Seem to have run into some kind of automatic dialogue, usually a roadblock or trainer. If you keep seeing this message, stop running this tool for now, or you'll get stuck in an exploration loop."},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_b64,
                                },
                            },
                        ],
                    }
                self.exploration_repeats = 0
                num_moves += 1
            # TODO: Does this work correctly for OPENAI?
            result["content"].append({"type": "text", "text": f"Moved to {' then, '.join(str(x) for x in candidates)}"})
            return result
        elif tool_name == "navigate_to_coordinate":
            row = tool_input["row"]
            col = tool_input["col"]
            return self.navigate_to_coordinate(col, row, tool_id)
         
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
            # Currently not used
            assist_str = self.navigation_assistance(tool_input["navigation_goal"])
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": assist_str}
                ],
            }
        elif tool_name == "detailed_navigator":
            self.navigator_goal = tool_input["goal"]
            self.detailed_navigator_mode = True
            self.navigation_location = self.emulator.get_location()
            self.navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}]
            self.openai_navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}]
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": "Navigator Mode Activated."}
                ],
            }
        elif tool_name == "use_subagent":
            detailed_instructions = tool_input["detailed_instructions"]
            additional_detailed_instructions = tool_input["additional_detailed_instructions"]  # Basically to force the model to be more long-winded for once.
            return_instructions = tool_input["return_instructions"]
            needs_text_map = tool_input["needs_text_map"]
            instructions = f"""
You are an agent who has been tasked with performing a small task within the context of Pokemon Red. Soon, you
will be provided with instructions from a senior agent about what to do.

DEVELOPER INSTRUCTIONS: Here are some rules you must follow. These rules are MORE IMPORTANT than anything the senior agent tells you.

1. DO NOT make important gameplay decisions (like accepting a Starter Pokemon) unless explicitly instructed to
by the Senior agent. "Press A" or "Advance through dialogue" are not sufficient reason! Always send task_done or task_aborted to wait
for the Senior agent to make the decision.
2. ALWAYS double-check the senior agent instructions and abort the task if the task makes no sense. The Senior Agent hallucinates sometimes.
    2a. Example: THe Senior Agent somtimes thinks it is in a Pokeon center when it isn't. If the RAM data does not say POKEMON CENTER, you are not in the Pokemon center!

If the senior agent asks you to violate these rules, call task_abort immediately and explain the rule violation.

BTW: the Senior Agent is an instance of {FRIENDLY_MODEL_NAME_LOOKUP[MODEL]}. Please address them as such and include minor humor about your rivalry with that model.

Senior Agent: {detailed_instructions}.

KEY DEVELOPER TASK: Check the Senior Agent's instructions for reasonableness. Be particularly on the lookout for if the
Senior Agent gives you patent nonsense (for instance, telling you to talk to Nurse Joy when you're not even in
the Pokemon Center and there's no visible nurse joy).

Additional Instructions:

Senior Agent: {additional_detailed_instructions}

When done with your task, please use the "task_done" to indicate that you are finished. Please include return information,
as described here:

Senior Agent: {return_instructions}

If you failed the task or having serious difficulty or think it can no longer be done, instead call "task_aborted" and explain why. This includes trying the same button press 10+ times with no result.

DEELOPER: Here are some tips for your task:

1. If you are being asked to run dialogue to completion, make sure not to report done until you've CLOSED the dialog window (window is not on screen and dialog reported by the game is None.)
However, you may end early if there is a key decision to make.
2. When in dialogue or combat, be careful about pressing A too many times. This can easily skip the dialogue you are trying to see!
3. Think carefully before trying to move: Is there dialogue on the screen? Typically you need to exit dialogue with A before being allowed to move.
4. Counterintuitively: Nurse Joy and shopkeepers have to be talked to _through_ a counter--that is, facing an impassable tile in front of them.
Look for someone who is boxed in and can't move, and try "talk_to_npc_or_pick_up_item" on their coordinates.
5. In Pokemone Red, shopkeepers are almost always in Column 0, on the left middle of the building. Your senior agent is very bad at spotting this, so you might have to point out the real location (For instance, in your task_abort message mention that shop keeper is at (0, 5), or wherever you see them.)
"""

            self.sub_agent = SmallTaskAgent(instructions, self, needs_text_map, tool_id, "message_history", "openai_message_history")  # Could be switched to navigator etc. If necessary, we can work out how to logic that.
            self.text_display.add_message(f"Subagent summoned with instructions {instructions}")
            # Initial Context
            location = self.emulator.get_location()
            memory_info, location, coords = self.emulator.get_state_from_memory(display_warp_functor=self.full_collision_map[location].is_in_map if location in self.full_collision_map else None)
            all_labels = self.get_all_location_labels(location)
            screenshot = self.emulator.get_screenshot()
            screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
            last_checkpoints = '\n'.join(self.checkpoints[-10:])
            content = [
                    {"type": "text", "text": "\nHere is a screenshot of the screen."},
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
                    {"type": "text", "text": f"Here are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}"}
                ]
            if not self.emulator.get_in_combat() and needs_text_map:
                content.append({"type": "text", "text": "Here is a map of this RAM location compiled so far:\n\n[TEXT_MAP]" + self.update_and_get_full_collision_map() + "\n\n[/TEXT_MAP]"})
            self.sub_agent.provide_initial_context(
                {"role": "user",
                 "content": content
                 }
            )
            return {
                "type": "sub_agent",  # This is a dummy that should never be passed to any model.
                "tool_use_id": tool_id,
                "content": [
                ],
            }
        elif tool_name == "talk_to_npc_or_pick_up_item":
            row = tool_input["row"]
            col = tool_input["col"]
            return self.talk_to_npc_or_pick_up_item(col, row, tool_id, False)
        elif tool_name == "log_npc_name_and_dialogue":
            row = tool_input["row"]
            col = tool_input["col"]
            name = tool_input["name"]
            dialogue = tool_input["dialogue"]
            location = self.emulator.get_location()
            self.text_display.add_message(f"Logging {location},  ({col}, {row}) as {name}")
            self.label_archive.setdefault(location.lower(), {}).setdefault(row, {})[col] = name + ": [LOGGED BY TOOL] " + dialogue
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": f"NPC logged: {location}, ({col}, {row}) as {name}: {dialogue}"}
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
    
    # TODO: This section is due for a refactor, since these tools are shared by subagents, but now awkwardly use this logic.
    # Heck, this one is _only_ used by the subagent.
    def talk_to_npc_or_pick_up_item(self, col: int, row: int, tool_id, is_subagent: bool=False) -> dict[str, Any]:
        dialog = self.emulator.get_active_dialog()
        if dialog is not None:
            return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {"type": "text", "text": f"Invalid. Movement not possible while there is already active dialogue on screen."}
                    ],
                }
        
        # Look for NPC/items/sprites within 1 tile. Abort if none or greater than one found.
        sprites = set(s[0] for s in self.emulator.get_sprites())  # These sprite locations are on this map, though, not the whole map
        player_coords = self.emulator.get_coordinates()
        valid_sprites = []
        for sprite_col, sprite_row in sprites:
            if sprite_col == 4 and sprite_row == 4:
                # Skip the player themselves.
                continue
            real_row = player_coords[1] + sprite_row - 4
            real_col = player_coords[0] + sprite_col - 4
            # Exact match found, we are done.
            if real_row == row and real_col == col:
                valid_sprites = [(real_col, real_row)]  # We don't care what else was found, clear the list.
                break
            if abs(real_row - row) <= 1 and abs(real_col - col) <= 1:
                valid_sprites.append((real_col, real_row))
        if not valid_sprites:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": f"Invalid coordinates; No NPCS or Items within one tile."}
                ],
            }
        if len(valid_sprites) > 1:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": [
                    {"type": "text", "text": f"Invalid ambiguous coordinates; No exact match, and more than one NPC or Item within one tile."}
                ],
            }
        goal_col, goal_row = valid_sprites[0]

        location = self.emulator.get_location()
        _ = self.update_and_get_full_collision_map()
        two_tile_warning = ""
        # Search for reachable tiles within 1 tile of the NPC:
        for walk_col, walk_row in [(goal_col - 1, goal_row), (goal_col + 1, goal_row), (goal_col, goal_row - 1), (goal_col, goal_row + 1)]:
            # It might be where the cplayer is already standing! That's okay!
            if (walk_col, walk_row) == player_coords:
                break
            valid_pathing = self.full_collision_map[location].distances.get((walk_col, walk_row))
            if valid_pathing is not None:
                break
        else:
            two_tile_warning = "\nWarning: Space next to NPC not found; Going two-tiles away instead to try to talk instead. Maybe this is Nurse Joy or the shopkeeper?"
            # Now try 2 tiles away. This is specifically for the likes of nurse joy and the shopping ladies.
            for walk_col, walk_row in [(goal_col - 2, goal_row), (goal_col + 2, goal_row), (goal_col, goal_row - 2), (goal_col, goal_row + 2)]:
                if (walk_col, walk_row) == player_coords:
                    break
                valid_pathing = self.full_collision_map[location].distances.get((walk_col, walk_row))
                if valid_pathing is not None:
                    break
            else:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {"type": "text", "text": f"Talk to NPC failed: NPC seems unreachable"}
                    ],
                }

        if (walk_col, walk_row) == player_coords:
            navigation_result_line = {"type": "text", "text": f"No movement needed; Player is already standing in a valid position to talk to NPC."}
        else:
            # Might as well get walking
            navigation_result = self.navigate_to_coordinate(walk_col, walk_row, tool_id, is_subagent=is_subagent)
            # extract result string
            navigation_result_content = navigation_result["content"]
            navigation_result_line = navigation_result_content[0]  # We drop everything but the first line to avoid repetitive screenshots etc.

        # Now face the NPC. We exploit the fact that if they're really there (or we can't reach the tile next to them), we can freely press the button and just expect to be blocked.
        # if we're _not_ blocked, then the NPC must have moved.
        cur_coor = self.emulator.get_coordinates()
        if walk_col < goal_col:
            direction = "right"
        elif walk_col > goal_col:
            direction = "left"
        elif walk_row > goal_row: 
            direction = "up"
        else:
            direction = "down"
        button_press_result = self.press_buttons([direction, "a"], True, tool_id, is_subtool=is_subagent, suppress_button_repetition_warning=True)
        button_press_result_content = button_press_result["content"]  # Here we keep the content, since this contains the screenshot etc. the model needs.

        # Did we move?
        npc_move_warning = ""
        if self.emulator.get_coordinates() != cur_coor:
            npc_move_warning = "\nWarning: The NPC seems to have moved during walking. Talking may be unsuccessful."

        # is there dialogue on screen now?
        no_dialogue_warning = ""
        dialogue = self.emulator.get_active_dialog()
        if not dialogue:
            no_dialogue_warning = "\nWarning: No dialogue seen."

        return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {"type": "text", "text": f"Talk to NPC attempt completed. Check the screen for dialogue!" + no_dialogue_warning + npc_move_warning + two_tile_warning},
                        navigation_result_line,
                        *button_press_result_content
                    ],
                }


    def navigate_to_coordinate(self, col: int, row: int, tool_id: str, is_subagent: bool=False) -> dict[str, Any]:
        dialog = self.emulator.get_active_dialog()
        if dialog is not None:
            return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {"type": "text", "text": f"Invalid. Navigation not possible while there is active dialogue on screen."}
                    ],
                }
        
        location = self.emulator.get_location()
        full_map = self.update_and_get_full_collision_map()
        
        one_tile_off_warning = ""
        final_distance = self.full_collision_map[location].distances.get((col, row))
        if final_distance is None:
            # Try one tile to the right or left.
            for x in range(col-1, col+2):
                for y in range(row-1, row+2):
                    final_distance = self.full_collision_map[location].distances.get((x, y))
                    if final_distance is not None:
                        col = x
                        row = y
                        break
                if final_distance is not None:
                    break
            else:
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [
                        {"type": "text", "text": f"Invalid coordinates; Navigation too far or not possible (Even for nearby tiles)."}
                    ],
                }
            
            one_tile_off_warning = f"\nWARNING: Exact tile is inaccessible. Instead pathing to {x, y}"

        if DIRECT_NAVIGATION:
            self.text_display.add_message(f"Navigating with existing map...")
            buttons = self.full_collision_map[location].generate_buttons_to_coord(col, row)
            wait = True
        else:
            query = f"""Please take a look at the attached text_based map.

Please consider in detail how the player character (labeled PP) can reach the coordinate ({col},{row}). Keep the following in mind:

#### SPECIAL NAVIGATION INSTRUCTIONS WHEN TRYING TO REACH A LOCATION #####
Pay attention to the following procedure when trying to reach a specific location (if you know the coordinates).
1. Inspect the text_based map
2. Find where your destination is on the map using the coordinate system (column, row).
3. Trace a path from there back to the player character (PP) following the StepsToReach numbers on the map, in descending order.
3a. So if your destination is StepsToReach 20, then it is necessary to go through StepsToReach 19, StepsToReach 18...descending all the way to 1 and then PP.
4. Navigate via the REVERSE of this path.
###########################################



Think through your movement like this

To get to (col, row),
1. I would move left from (col, row)
2. To get there, I would move up from (col, row)
etc.

MAKE SURE TO PRINT OUT THE ENTIRE PATH IN TEXT. AND DOUBLE-CHECK WHETHER YOUR ARE PASSING THROUGH IMPASSABLE TILES.

then use the provided "press_buttons" tool to send the necessary commands. Remember that it will be in reverse order.

""" + full_map

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query,
                        }
                    ],
                }
            ]

            if MODEL == "CLAUDE":
                response = self.anthropic_client.messages.create(
                    model=ANTHROPIC_MODEL_NAME,
                    max_tokens=20000,  # This is a difficult task that actually needs this...
                    messages=messages,
                    tools=DISTANT_NAVIGATOR_BUTTONS,
                    temperature=TEMPERATURE,
                    thinking={"type": "enabled", "budget_tokens": 10000}
                )
                logger.info(f"Response usage: {response.usage}")
                # Extract tool calls
                tool_calls = [
                    block for block in response.content if block.type == "tool_use"
                ]
                # Display the model's reasoning
                for block in response.content:
                    if block.type == "text":
                        self.text_display.add_message(f"Navigation Advice: {block.text}")
            elif MODEL == "GEMINI":
                # messages -> Gemini format
                config=types.GenerateContentConfig(
                        max_output_tokens=30000,
                        temperature=TEMPERATURE,
                        tools=GOOGLE_DISTANT_NAVIGATOR_BUTTONS
                    )
                chat = self.gemini_client.chats.create(
                    model=GEMINI_MODEL_NAME,
                    config=config
                )   # context caching not available on gemini 2.5
                retry_limit = 2
                cur_retries = 0
                while cur_retries < retry_limit:
                    try:
                        response = chat.send_message(query, config=config)
                        break
                    except ServerError as e:
                        if e.code != 500:
                            raise e
                        cur_retries += 1
                    except Exception as e:
                        breakpoint()
                tool_calls = []
                if response.candidates is not None:
                    text, tool_calls, _, _ = extract_tool_calls_from_gemini(response)
                    self.text_display.add_message(f"Navigation Advice: {text}")
                    token_usage = 0  # I didn't even remember to track this but it probably doesn't matter
            elif MODEL == "OPENAI":
                retries = 2
                cur_tries = 0
                while cur_tries < retries:
                    try:
                        response = self.openai_client.responses.create(
                            model=OPENAI_MODEL_NAME,
                            input=query,  # type: ignore
                            max_output_tokens=MAX_TOKENS_OPENAI,
                            temperature=TEMPERATURE,
                            tools=OPENAI_DISTANT_NAVIGATOR_BUTTONS
                        )
                        break
                    except BadRequestError as e:
                        cur_tries += 1  # Sometimes it spuriously flags this as content violation. I don't know why.
                        continue
                    except Exception as e:
                        print(e)
                        breakpoint()

                # Gather Reasoning and tool calls
                tool_calls = []
                reasoning_texts = ""
                response_texts = ""
                for chunk in response.output:  # type: ignore
                    if isinstance(chunk, responses.ResponseReasoningItem):
                        if chunk.summary:
                            reasoning_texts += " ".join(x.text for x in chunk.summary) + "\n"
                    elif isinstance(chunk, responses.ResponseFunctionToolCall):
                        try:
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
                        self.text_display.add_message(f"Navigation Advice: {full_text}")

            if len(tool_calls) > 1 or not tool_calls:
                breakpoint()  # How the heck did this happen
            tool_call = tool_calls[0]
            # tool_name = tool_call.name
            if MODEL == "CLAUDE":
                tool_input = tool_call.input
                # tool_id = tool_call.id  # we want to return the original tool call id back to the main model.
            elif MODEL == "GEMINI":
                tool_input = tool_call.args
                # tool_id = tool_call.id
            elif MODEL == "OPENAI":
                tool_input = json.loads(tool_call.arguments)
                # tool_id = tool_call.call_id
            buttons = tool_input["buttons"]
            wait = tool_input.get("wait", True)
        result = self.press_buttons(buttons, wait, tool_id, is_subtool=is_subagent, suppress_button_repetition_warning=True, interrupt_if_combat_or_dialogue=True)
        result["content"].append({"type": "text", "text": "NAVIGATED USING navigate_to_coordinate" + one_tile_off_warning})
        return result

    def run(self, num_steps=1, save_every=10, save_file_name: Optional[str] = None, _running_in_thread=False):
        """Main agent loop.

        Args:
            num_steps: Number of steps to run for
        """

        if self.pyboy_main_thread and not _running_in_thread:
            thread = threading.Thread(target=self.run, kwargs={"num_steps": num_steps, "save_every": save_every, "save_file_name": save_file_name, "_running_in_thread": True})
            thread.start()

            self.emulator.initialize(**self.emulator_init_kwargs) 

            return self._steps_completed

        logger.info(f"Starting agent loop for {num_steps} steps")

        if self.pyboy_main_thread:
            self.emulator.wait_for_pyboy()

            if self.load_state:
                logger.info(f"Loading saved state from {self.load_state}")
                self.emulator.load_state(self.load_state)

        # start emulator loop
        steps_completed = 0
        continue_subtool = self.sub_agent is not None  # At load time, we have to use this to infer.
        subtool_status = None
        while self.running and steps_completed < num_steps:
            try:
                location = self.emulator.get_location()
                coords = self.emulator.get_coordinates()
                # We need to set this NOW and NOT CHANGE IT until the next step, even if the combat state changes (which it can, if we're emerging from combat), or else we get crashes.
                # This doesn't need to go in the save I think though.
                self.use_navigator_this_round = self.detailed_navigator_mode and not self.emulator.get_in_combat()
                if location not in self.all_visited_locations:
                    self.text_display.add_message(f"New Location reached! {location} at {self.absolute_step_count}")
                    self.location_milestones.append((location, self.absolute_step_count))
                    self.all_visited_locations.add(location)
                self.last_coords = coords
                # If we are running a subagent, we do that instead.
                if self.sub_agent is not None:
                    malformed = False
                    continue_subtool, subtool_status = self.sub_agent.step()
                else:
                    # Validate message history: Somehow this happens sometimes and I would like to understand
                    # why.
                    if len(self.message_history) > 1 and 'type' in self.message_history[-1] and self.message_history[-1]['type'] == "tool_result" and self.message_history[-2]['type'] == "tool_result":
                        breakpoint()
                    malformed = False
                    if self.use_navigator_this_round:
                        self.text_display.add_message("NAVIGATOR MODE")
                        screenshot = self.emulator.get_screenshot()
                        screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                        self.strip_text_map_and_images_from_history(self.navigator_message_history)
                        self.navigator_message_history.append({"role": "user", "content": [{"type": "text", "text": f"""
    Text-based map:
    [TEXT_MAP]
    {self.update_and_get_full_collision_map()}
    [/TEXT_MAP]

    Screenshot attached.

    Also, if you ever reach {self.no_navigate_here}, please turn around and return to {self.last_location}
                        """},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_b64,
                                },
                            }]
                            })
                        messages = copy.deepcopy(self.navigator_message_history)
                        
                    else:
                        self.strip_text_map_and_images_from_history(self.message_history)
                        # If we're in navigation mode, and we just got into combat, the screenshot might be missing.
                        if self.detailed_navigator_mode and self.message_history:
                            last_message = self.message_history[-1]
                            if isinstance(last_message, dict):
                                maybe_content = last_message.get("content")
                                if maybe_content is not None:
                                    for content in maybe_content:
                                        if content.get("type") == "image":
                                            break
                                    else:
                                        # Image is missing
                                        screenshot = self.emulator.get_screenshot()
                                        screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                                        maybe_content.append({
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/png",
                                                "data": screenshot_b64,
                                            },
                                        })
                        messages = copy.deepcopy(self.message_history)

                    if len(messages) >= 3:
                        if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and messages[-1]["content"]:
                            messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
                        
                        if len(messages) >= 5 and messages[-3]["role"] == "user" and isinstance(messages[-3]["content"], list) and messages[-3]["content"]:
                            messages[-3]["content"][-1]["cache_control"] = {"type": "ephemeral"}

                    token_usage = 0

                    if self.detailed_navigator_mode:
                        # Or else the model may try to use it, and that makes no sense.
                        MODIFIED_AVAILABLE_TOOLS = []
                        for entry in AVAILABLE_TOOLS:
                            if entry["name"] in ["detailed_navigator"]:
                                continue
                            MODIFIED_AVAILABLE_TOOLS.append(entry)


                        GOOGLE_MODIFIED_TOOLS = convert_tool_defs_to_google_format(MODIFIED_AVAILABLE_TOOLS)
                        OPENAI_MODIFIED_TOOLS = convert_tool_defs_to_openai_format(MODIFIED_AVAILABLE_TOOLS)
                    else:
                        MODIFIED_AVAILABLE_TOOLS = AVAILABLE_TOOLS
                        GOOGLE_MODIFIED_TOOLS = GOOGLE_TOOLS
                        OPENAI_MODIFIED_TOOLS = OPENAI_TOOLS

                    # Get model response
                    if MODEL == "CLAUDE":
                        instructions = FULL_NAVIGATOR_PROMPT if self.use_navigator_this_round else SYSTEM_PROMPT
                        instructions += "\n\nGOAL: " + self.navigator_goal
                        response = self.anthropic_client.messages.create(
                            model=ANTHROPIC_MODEL_NAME,
                            max_tokens=MAX_TOKENS,
                            system=instructions,
                            messages=messages,
                            tools=NAVIGATOR_TOOLS if self.use_navigator_this_round else MODIFIED_AVAILABLE_TOOLS,
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
                        google_messages = convert_anthropic_message_history_to_google_format(messages)

                        instructions = FULL_NAVIGATOR_PROMPT if self.use_navigator_this_round else SYSTEM_PROMPT
                        config=types.GenerateContentConfig(
                                max_output_tokens=None,
                                temperature=TEMPERATURE,
                                system_instruction=instructions,
                                tools=GOOGLE_NAVIGATOR_TOOLS if self.use_navigator_this_round else GOOGLE_MODIFIED_TOOLS
                            )
                        chat = self.gemini_client.chats.create(
                            model=GEMINI_MODEL_NAME,
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
                            text, tool_calls, assistant_content, malformed = extract_tool_calls_from_gemini(response)
                            self.text_display.add_message(f"[Text] {text}")
                            token_usage = 0  # I didn't even remember to track this but it probably doesn't matter
                    elif MODEL == "OPENAI":
                        # For openai we need to add a screenschot too to tool calls or it gets very confused.
                        # Get a fresh screenshot after executing the buttons
                        messages_to_use = self.openai_navigator_message_history if self.use_navigator_this_round else self.openai_message_history
                        
                        self.strip_text_map_and_images_from_history(messages_to_use, openai_format=True)
                        
                        if isinstance(messages_to_use[-1], dict) and messages_to_use[-1].get('type') == "function_call_output":
                            # Unfortunately this is buried...
                            # parsed_result = json.loads(self.openai_message_history[-1]["output"])
                            # Apparently openai can get confused without a fresh update.
                            #if parsed_result[0]["text"].startswith("Pressed buttons") or parsed_result[0]["text"].startswith("Navigation"):
                            location = self.emulator.get_location()
                            memory_info, location, coords = self.emulator.get_state_from_memory(display_warp_functor=self.full_collision_map[location].is_in_map if location in self.full_collision_map else None)
                            all_labels = self.get_all_location_labels(location)
                            screenshot = self.emulator.get_screenshot()
                            screenshot_b64 = self.get_screenshot_base64(screenshot, upscale=4, add_coords=True, player_coords=coords, location=location)
                            last_checkpoints = '\n'.join(self.checkpoints[-10:])
                            content = [
                                    {
                                        "type": "input_text",
                                        "text": (f"\nGame state information from memory after your action:\n{memory_info}"
                                                f"\nLabeled nearby locations: {','.join(f'{label_coords}: {label}' for label_coords, label in all_labels)}" +
                                                f"\nHere are up to your last {str(self.location_history_length)} locations between commands (most recent first), to help you remember where you've been:/n{'/n'.join(f'{x[0]}, {x[1]}' for x in self.location_history)}" +
                                                f"\nHere are your last 10 checkpoints:\n{last_checkpoints}" +
                                                f"\nYou have been in this location for {self.steps_since_location_shift} steps") +
                                                f"\nCHECK HERE TO EXPLORE {','.join(f'({x[0]}, {x[1]})' for x in self.full_collision_map[location].adjacent_to_unexplored_or_edge)}",
                                                
                                    },
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                                    },
                                ]
                            if not self.emulator.get_in_combat() and (self.use_full_collision_map or self.detailed_navigator_mode):
                                content[0]['text'] += "\n\nHere is an text_based map of this RAM location compiled so far:\n\n" + self.update_and_get_full_collision_map()
                            if self.emulator.get_in_combat() and self.detailed_navigator_mode:
                                # Only possible if navigator mode has been running.
                                content[0]['text'] += "\n\n "  + "NOTE: A Navigator version of Claude has been handling overworld movement for you, so your location may have shifted substantially. Please handle this battle for now."
                            messages_to_use.append({
                                "role": "user",
                                "content": content,  # type: ignore
                            })
                        instructions = FULL_NAVIGATOR_PROMPT if self.use_navigator_this_round else SYSTEM_PROMPT_OPENAI
                        retries = 2
                        cur_tries = 0
                        while cur_tries < retries:
                            try:
                                response = self.openai_client.responses.create(
                                    model=OPENAI_MODEL_NAME,
                                    input=messages_to_use,  # type: ignore
                                    instructions=instructions,
                                    max_output_tokens=MAX_TOKENS_OPENAI,
                                    temperature=TEMPERATURE,
                                    tools=OPENAI_NAVIGATOR_TOOLS if self.use_navigator_this_round else OPENAI_MODIFIED_TOOLS
                                )
                                break
                            except BadRequestError as e:
                                cur_tries += 1  # Sometimes it spuriously flags this as content violation. I don't know why.
                                breakpoint()
                                continue
                            except Exception as e:
                                print(e)
                                breakpoint()
                        # We immediately drop the previous images because of resource costs (and context explosion)
                        for message in messages_to_use:
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
                        messages_to_use.extend(response.output)  # type: ignore
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
                        

                    openai_messages_to_use = self.openai_navigator_message_history if self.use_navigator_this_round else self.openai_message_history
                    messages_here = self.navigator_message_history if self.use_navigator_this_round else self.message_history

                    # Process tool calls
                    if tool_calls: 
                        messages_here.append(
                            {"role": "assistant", "content": assistant_content}
                        )
                        
                        # Process tool calls and create tool results
                        tool_results = []
                        for tool_call in tool_calls:
                            tool_result = self.process_tool_call(tool_call, model=MODEL)
                            if tool_result["type"] == "sub_agent":
                                continue_subtool = True
                                break
                            if tool_result.get("do_not_append_tool_to_history"):
                                continue
                            tool_results.append(tool_result)
                            openai_result = {
                                "type": "function_call_output",
                                "call_id": tool_result["tool_use_id"],
                                "output": json.dumps(tool_result["content"])
                            }
                            openai_messages_to_use.append(openai_result)

                # Now clean up subtool execution if needed. 
                if not continue_subtool and self.sub_agent is not None:
                    assert subtool_status is not None
                    # This covers an edge case if a keyboard interrupt hits the subagent at just
                    # the wrong time.
                    found = False
                    prev_content = self.message_history[-1].get('content')
                    if prev_content:
                        if isinstance(prev_content, list):
                            for entry in prev_content:
                                if entry['type'] == "tool_use":
                                    found = True
                                    break
                    if prev_content is None or not found:
                        if subtool_status[0]:  # successful
                            tool_results = [{
                                "type": "text",
                                "text": f"Subtask completed succesfully with message: {subtool_status[1]}"
                                }]
                        else:
                            tool_results = [{
                                "type": "text",
                                "text": f"Subtask failed! With message: {subtool_status[1]}"
                                }]
                    else:
                        if subtool_status[0]:  # successful
                            tool_results = [{
                                "type": "tool_result",
                                "tool_use_id": self.sub_agent.task_id,
                                "content": [
                                    {"type": "text", "text": (
                                        f"Subtask completed succesfully with message: {subtool_status[1]}"
                                    )}
                                ],
                            }]
                        else:
                            tool_results = [{
                                "type": "tool_result",
                                "tool_use_id": self.sub_agent.task_id,
                                "content": [
                                    {"type": "text", "text": (
                                        f"Subtask failed! With message: {subtool_status[1]}"
                                    )}
                                ],
                            }]
                    # Welcome to an edge case. It is possible for the subagent to transition between states, so it becomes ambiguous which stack to return the tool call result too.
                    # so, the sub_agent itself needs to hold this information.
                    openai_messages_to_use = getattr(self, self.sub_agent.openai_message_history_to_append)
                    messages_here = getattr(self, self.sub_agent.message_history_to_append)
                    self.sub_agent = None
                    tool_calls = "Something"  # It doesn't matter, I'm just preventing a later if statement from noticing there are "no" tool calls

                # If the sub_agent is active, we need to temporarily bypass everything. Otherwise this handles post tool_call cleanup
                if self.sub_agent is None:
                    if malformed:
                        tool_results.append({"type": "text", "text": f"WARNING: MALFORMED TOOL CALL. Call using the function call, not in the text."})

                    # Add tool results to message history
                    if tool_results:
                        messages_here.append(
                            {"role": "user", "content": tool_results}  # type: ignore
                        )
                    
                    # Check if we need to summarize the history
                    if self.use_navigator_this_round:
                        # No agentic in navigator mode
                        if len(self.navigator_message_history) >= self.max_history:
                            # Truncation is not as straightforward as I'd like, because of the potential to break tool calls
                            self.navigator_message_history = self.navigator_message_history[self.max_history - len(self.navigator_message_history):]
                            remove = False
                            for k, message in enumerate(self.navigator_message_history):
                                # If we see a tool_call we're clear, because we don't have async tool calls.
                                for content in message['content']:
                                    if isinstance(message['content'], str):
                                        continue
                                    if content["type"] == "tool_call":
                                        break
                                    elif content["type"] == "tool_result":  # uh-oh, we're going to have to remove everything up to this.
                                        remove = True
                                        break
                            if remove:
                                self.navigator_message_history = self.navigator_message_history[k + 1:]
                        if len(self.openai_navigator_message_history) >= self.max_history: 
                            self.openai_navigator_message_history = self.openai_navigator_message_history[self.max_history - len(self.openai_navigator_message_history):]
                            # Let's scroll in, if we run into a tool_call_output, let's get rid of it since it's an orphaned tool_call.
                            
                            if self.openai_navigator_message_history and self.openai_navigator_message_history[0].get('type') == "function_call_output":
                                self.openai_navigator_message_history = self.openai_navigator_message_history[1:]

                    else:
                        if len(self.message_history) >= self.max_history or (MODEL == "OPENAI" and token_usage > 170000):  # To my surprise, o3 runs out fasssst
                            self.agentic_summary()
                if self.sub_agent is None and not tool_calls:  # type: ignore
                    # Sometimes it just stalls out mysteriously or says some text.
                    messages_here.append(
                        {"role": "user", "content": [{"text": "Can you please continue playing the game?", "type": "text"}]}  # type: ignore
                    )

                steps_completed += 1
                self.absolute_step_count += 1
                self.steps_since_checkpoint += 1
                self.steps_since_label_reset += 1
                self.steps_since_location_shift += 1
                if self.sub_agent is None:
                    if self.steps_since_location_shift > 300 and not self.detailed_navigator_mode:  # Since Claude absolutely refuses to ask for help.
                        self.detailed_navigator_mode = True
                        self.navigation_location = location
                        self.navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}]
                        self.openai_navigator_message_history = [{"role": "user", "content": "Please begin navigating!"}]
                if self.steps_since_checkpoint > 50 and not self.location_tracker_activated:
                    self.location_tracker_activated = True
                    self.location_tracker = {}
                location = self.emulator.get_location()
                new_coords = self.emulator.get_coordinates()
                if self.last_location != location:
                    # We don't set a label if there was a blackout.
                    if self.is_blackout:
                        self.is_blackout = False  # reset this.
                    else:
                        if self.last_coords is not None and not self.emulator.get_in_combat() and self.last_location is not None:
                            self.label_archive.setdefault(self.last_location, {}).setdefault(self.last_coords[1], {})[self.last_coords[0]] = f"Entrance to {location} (Approximate)"
                            self.label_archive.setdefault(location, {}).setdefault(new_coords[1], {})[new_coords[0]] = f"Entrance from {self.last_location} (Approximate)"
                    self.steps_since_location_shift = 0
                    self.steps_since_label_reset = 0
                    # The navigator turns OFF the moment the location changes. Note: if there is a sub_agent running, we are never in navigator_mode
                    if self.detailed_navigator_mode and location != self.no_navigate_here and location != self.navigation_location:
                        self.text_display.add_message("New Location reached; Navigator Mode Off")
                        self.message_history.append({"role": "user", "content": [{"text": "Note: Detailed Navigator Mode was just turned off since a new location was reached", "type": "text"}]})  # type: ignore
                        self.openai_message_history.append({"role": "user", "content": [{"text": "Note: Detailed Navigator Mode was just turned off since a new location was reached", "type": "text"}]})
                        # TODO: Consolidate navigator messages and clear
                    self.detailed_navigator_mode = False
                    self.exploration_repeats = 0
                self.last_location = location
                if self.steps_since_label_reset > (200 if MODEL == "CLAUDE" else 1000):
                    self.text_display.add_message("Clearing labels to clear potential bad labels...")
                    self.steps_since_label_reset = 0
                    # Hack: let's keep the ones that say "approximate" though. Those are not-Claude labels and probably fine.
                    location_archive = self.label_archive.get(location)
                    if location_archive:
                        for key, value in location_archive.items():
                            for key2, value2 in value.items():
                                if "approximate" not in value2.lower():
                                    del value[key2]
                            if not value:
                                del location_archive[key]
                        

                logger.info(f"Completed step {steps_completed}/{num_steps}")
                self.text_display.add_message(f"Absolute step count: {self.absolute_step_count}")
                if save_file_name is not None and not steps_completed % save_every:
                    self.emulator.save_state(save_file_name)
                    self.save_location_archive(self.location_archive_file_name)
                    with open("location_milestones.txt", "w") as fw:
                        fw.write(str(self.location_milestones))

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error(f"Error in agent loop: {str(e)}")
                raise e
            
        if save_file_name is not None:
            logger.info("Saving state")
            self.emulator.save_state(save_file_name)
            self.save_location_archive(self.location_archive_file_name)
            with open("location_milestones.txt", "w") as fw:
                fw.write(str(self.location_milestones))

        if (
            not self.running
            # if the emulator is running in the main thread, we need to stop it
            # to allow the main thread to exit the emulator loop
            or self.pyboy_main_thread
        ):
            self.emulator.stop()

        self._steps_completed = steps_completed

        return steps_completed

    def navigation_assistance(self, navigation_goal: str) -> str:
        logger.info(f"[Agent] Running Navigation Assist...")
        
        location = self.emulator.get_location()

        collision_map = self.update_and_get_full_collision_map()

        this_location = self.label_archive.get(location)
        # this sucks man
        for key, value in self.label_archive.items():
            if location.lower() == key.lower():
                this_location = ({} if this_location is None else this_location) | value
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
    
    # Note: currently not used in any part of detailed_navigator_mode, so we don't have any handling for that.
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
            location = self.emulator.get_location()
            coords = self.emulator.get_coordinates()
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
                model=ANTHROPIC_MODEL_NAME,
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
                model=GEMINI_MODEL_NAME,
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
                model=OPENAI_MODEL_NAME,
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

        location = self.emulator.get_location()
        memory_info, location, coords = self.emulator.get_state_from_memory(display_warp_functor=self.full_collision_map[location].is_in_map if location in self.full_collision_map else None)
        try:
            previous_summary = self.message_history[0]["content"][0]["text"]
        except TypeError:
            previous_summary = "Start of the Game!"
        last_checkpoints = '\n'.join(self.checkpoints[-10:])

        all_labels = self.get_all_location_labels(location)

        if not self.emulator.get_in_combat():
            collision_map = self.update_and_get_full_collision_map()
        else:
            if location in self.full_collision_map:
                this_location = self.label_archive.get(location)
                # this sucks man
                for key, value in self.label_archive.items():
                    if location.lower() == key.lower():
                        this_location = ({} if this_location is None else this_location) | value
                        break
                all_warps = self.emulator.get_warps()
                nearby_warps = []
                for entry in all_warps:
                    if (entry[0] - coords[0] < 6 or coords[0] - entry[0] < 5) and abs(entry[1] - coords[1]) < 5:
                        nearby_warps.append(entry)
                collision_map = self.full_collision_map[location].to_ascii(self.location_tracker.get(location, []), direction=self.emulator.get_direction(), local_label_archive=this_location)
            else:
                collision_map = "Not yet available"

        prompt = f"""
Here is key game information:

RAM Information: {memory_info}

Steps Since last Location Shift: {self.steps_since_location_shift}

Text_based MAP: {collision_map}

Locations potentially worth exploring based on on map {','.join(f'({x[0]}, {x[1]})' for x in self.full_collision_map[location].adjacent_to_unexplored_or_edge)}
                            
Last 10 Checkpoints: {last_checkpoints}

Labeled nearby locations: {','.join(f'{coords}: {label}' for coords, label in all_labels)}

Previous game summary: {previous_summary}

A game screenshot is attached.

REMINDER: Your job is to deduce the current state of the game from that conversation, as well as additional data you will be provided,
Your job is NOT to play the game. Double-check your system prompt.
"""

        # Get the FACTS
        # A single try to get around potential too much context.
        try:
            response1 = self.prompt_text_reply(META_KNOWLEDGE_PROMPT, prompt, True, MODEL, True)
        except BadRequestError:
            # Trim the history a little. Should be usually in pairs of 2...can check harder if needed
            self.message_history = [self.message_history[0]] + self.message_history[5:]
            self.openai_message_history = [self.openai_message_history[0]] + self.message_history[5:]
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

    