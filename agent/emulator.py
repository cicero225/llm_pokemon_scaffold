import io
import logging
import numpy as np
import pickle
from collections import deque
import heapq
import random
import time
import threading, queue

from agent.memory_reader import PokemonRedReader, StatusCondition
from PIL import Image
from pyboy import PyBoy
from typing import Optional, Callable

logger = logging.getLogger(__name__)


# Thanks Gemini for letting me not have to think about this.
class PriorityLock:
    def __init__(self):
        self._lock = threading.Lock()
        self._queue = queue.PriorityQueue()

    def acquire(self, priority):
        self._queue.put((priority, threading.get_ident()))
        with self._lock:
            _, current_thread = self._queue.get()
            if current_thread != threading.get_ident():
                raise RuntimeError("Priority lock acquired out of order")

    def release(self):
        with self._lock:
            self._queue.task_done()

    # Annoying but necessary for neat with statements with argument.
    class PriorityLockContextHandler:
        def __init__(self, priority_lock, priority):
            self.priority_lock = priority_lock
            self.priority = priority

        def __enter__(self):
            self.priority_lock.acquire(self.priority)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.priority_lock.release()
            return False

    def __call__(self, priority):
        return self.PriorityLockContextHandler(self, priority)
    
    


class Emulator:
    def __init__(self):
        self.first_run = True
        self.run_thread: threading.Thread
        self.button_queue: queue.Queue
        self.pyboy_lock = PriorityLock()
        self.button_queue_clear = threading.Event()

        # Dialogue handler (for keeping track of multiple lines of dialogue e.g. in combat)
        self.dialogue_thread: threading.Thread
        self.stored_dialogue = []
        self.is_dialogue_running = False
        self.dialogue_lock = threading.Lock()

    # pyboy just doesn't work unless it's ticking and receiving button presses on the same thread as it was initialoized on, so
    # if we want it to be able to run independently we have to resort to keeping it in its own little box like this (in a thread)
    def player(self, rom_path, headless=True, sound=False):
        self.button_queue = queue.Queue()

        if headless:
            self.pyboy = PyBoy(
                rom_path,
                window="null",
                cgb=True,
            )
        else:
            self.pyboy = PyBoy(
                rom_path,
                cgb=True,
                sound=sound,
            )
        # Run the emulator for a short time to make sure it's ready
        self.pyboy.set_emulation_speed(0)
        for _ in range(60):
            self.pyboy.tick(1)
        self.pyboy.set_emulation_speed(1)
        while True:
            with self.pyboy_lock(10):
                try:
                    # terrible hack. Maybe I should just do a function passing interface...
                    item, press_or_release = self.button_queue.get(block=False)
                    if item == "wait":
                        for _ in range(press_or_release):
                            if not self.pyboy.tick(1):
                                if self.button_queue.empty():
                                    self.button_queue_clear.set()
                                return
                    elif item == "load_state":
                        self.pyboy.load_state(press_or_release)
                    elif item == "save_state":
                        self.pyboy.save_state(press_or_release)
                    elif item == "stop":
                        self.pyboy.stop()
                        self.button_queue_clear.set()
                        return
                    else:
                        if press_or_release:
                            self.pyboy.button_press(item)
                        else:
                            self.pyboy.button_release(item)
                except queue.Empty:
                    if not self.pyboy.tick(1):
                        self.button_queue_clear.set()
                        return
                except KeyboardInterrupt:
                    self.pyboy.stop()
                    self.button_queue_clear.set()
                    return

                if self.button_queue.empty():
                    self.button_queue_clear.set()

    """def tick(self, frames):
        # Advance the emulator by the specified number of frames.
        for _ in range(frames):
            self.pyboy.tick()"""

    def initialize(self, rom_path, headless=True, sound=False, pyboy_main_thread=False):
        """Initialize the emulator."""
        if pyboy_main_thread:
            self.player(rom_path=rom_path, headless=headless, sound=sound)
        else:
            self.run_thread = threading.Thread(target=self.player, kwargs={"rom_path": rom_path, "headless": headless, "sound": sound})
            self.run_thread.start()
            self.wait_for_pyboy()
            self.begin_updating_dialogue()

    def wait_for_pyboy(self):
        while True:
            try:
                self.pyboy # geh
            except AttributeError:
                time.sleep(0.1)
            else:
                break

    def begin_updating_dialogue(self):  # Call after wait_for_pyboy
        # A crass way of doing this, but not easy to do otherwise without breaking into pyboy.
        self.is_dialogue_running = True
        dialogue_check_interval = 0.2
        def dialogue_update():
            while self.is_dialogue_running:
                cur_dialogue = self.get_active_dialog()
                with self.dialogue_lock:
                    if cur_dialogue is None:  # We might lose a few lines at the end of combat or something but this will prevent problems.
                        self.stored_dialogue.clear()
                    else:
                        # If we have two instances of identical dialogue except one has a ►, favor the one with the arrow. Otherwise favor the more recent one.
                        if self.stored_dialogue and cur_dialogue.replace("▼", "").replace("►", "").strip() == self.stored_dialogue[-1].replace("▼", "").replace("►", "").strip():
                            if "►" in cur_dialogue:  # If this is not the case we should keep stored_dialogue.  
                                self.stored_dialogue[-1] = cur_dialogue
                        elif self.stored_dialogue and cur_dialogue.startswith(self.stored_dialogue[-1]):  # we have overlapping dialogue
                            self.stored_dialogue[-1] = cur_dialogue  # There's a number of edge cases that might hit this but for the sake of simplicity we're ignoring those.
                            # Now we must compared -1 and -2 if possible since they might match now. If so we need to wipe the last entry to prevent stutter.
                            if len(self.stored_dialogue) > 1 and self.stored_dialogue[-1].replace("▼", "").replace("►", "").strip() == self.stored_dialogue[-2].replace("▼", "").replace("►", "").strip():
                                self.stored_dialogue.pop()
                        else:
                            self.stored_dialogue.append(cur_dialogue)
                time.sleep(dialogue_check_interval)
        self.dialogue_thread = threading.Thread(target=dialogue_update)
        self.dialogue_thread.start()

    def get_stored_dialogue(self):
        if not self.stored_dialogue:
            return None
        with self.dialogue_lock:
            all_dialogue = "\n".join(self.stored_dialogue)
            self.stored_dialogue.clear()
        return all_dialogue

    def get_screenshot(self):
        """Get the current screenshot. We wait for the queue to clear to make sure all buttons are pressed."""
        self.button_queue_clear.wait()
        time.sleep(0.2 + random.random())  # this is just a good idea to wait for things to happen. Some randomization fixes arcane cases.
        return Image.fromarray(self.pyboy.screen.ndarray)

    def load_state(self, state_filename):
        """
        Load a state from a pickled file into the emulator.
        The pickled file should contain a dictionary with a 'pyboy_state' key.
        
        Args:
            state_filename: Path to the state file
        """
        # self.pyboy.load_state(open(state_filename, "rb"))
        with self.pyboy_lock(1):  # We really need to get this in correctly.
            self.button_queue_clear.clear()
            self.button_queue.put(("load_state", open(state_filename, "rb")))
        self.button_queue_clear.wait()
        logger.info("State loaded successfully.")

    def save_state(self, state_filename):
        with self.pyboy_lock(1):  # We really need to get this in correctly.
            self.button_queue_clear.clear()
            self.button_queue.put(("save_state", open(state_filename, "wb")))
        self.button_queue_clear.wait()
        logger.info("State saved successfully.")

    def press_buttons(self, buttons, wait=True, wait_for_finish=True) -> tuple[str, tuple[int, int]]:
        """Press a sequence of buttons on the Game Boy.
        
        Args:
            buttons (list[str]): List of buttons to press in sequence
            wait (bool): Whether to wait after each button press
            
        Returns:
            str: result of button press
            coords: the second-to-last coordinate visited, for tracking purposes.
        """
        results = []
        
        last_coords = self.get_coordinates()
        next_coords = last_coords
        for button in buttons:
            if button not in ["a", "b", "start", "select", "up", "down", "left", "right"]:
                results.append(f"Invalid button: {button}")
                continue
            
            with self.pyboy_lock(1):
                if wait_for_finish:
                    self.button_queue_clear.clear()
                # self.pyboy.button_press(button)
                self.button_queue.put((button, True))
                # self.tick(10)   # Press briefly
                self.button_queue.put(("wait", 10))
                # self.pyboy.button_release(button)
                self.button_queue.put((button, False))
                
                if wait:
                    # self.tick(120) # Wait longer after button release
                    if button in ["a", "b"]:
                        self.button_queue.put(("wait", 1000))
                    else:
                        self.button_queue.put(("wait", 250))
                else:
                    # self.tick(10)   # Brief pause between button presses
                    self.button_queue.put(("wait", 120))
            if wait_for_finish:
                self.button_queue_clear.wait()
                
            results.append(f"Pressed {button}")
            cur_coords = self.get_coordinates()
            if cur_coords != next_coords:
                last_coords = next_coords
                next_coords = cur_coords
        
        return "\n".join(results), last_coords

    def get_coordinates(self):
        """
        Returns the player's current coordinates from game memory.
        Returns:
            tuple[int, int]: (x, y) coordinates
        """
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_coordinates()

    def get_active_dialog(self):
        """
        Returns the active dialog text from game memory.
        Returns:
            str: Dialog text
        """
        reader = PokemonRedReader(self.pyboy.memory)
        dialog = reader.read_dialog()
        if dialog:
            return dialog
        return None

    def get_location(self):
        """
        Returns the player's current location name from game memory.
        Returns:
            str: Location name
        """
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_location()

    def _get_direction(self, array):
        """Determine the player's facing direction from the sprite pattern."""
        # Look through the array for any 2x2 grid containing numbers 0-3
        rows, cols = array.shape

        for i in range(rows - 1):
            for j in range(cols - 1):
                # Extract 2x2 grid
                grid = array[i : i + 2, j : j + 2].flatten()

                # Check for each direction pattern
                if list(grid) == [0, 1, 2, 3]:
                    return "down"
                elif list(grid) == [4, 5, 6, 7]:
                    return "up"
                elif list(grid) == [9, 8, 11, 10]:
                    return "right"
                elif list(grid) == [8, 9, 10, 11]:
                    return "left"

        return "no direction found"

    def _downsample_array(self, arr: np.ndarray) -> np.ndarray:
        """Downsample an 18x20 array to 9x10 by averaging 2x2 blocks."""
        # Ensure input array is 18x20
        if arr.shape != (18, 20):
            raise ValueError("Input array must be 18x20")

        # Reshape to group 2x2 blocks and take mean
        return arr.reshape(9, 2, 10, 2).mean(axis=(1, 3))

    def get_collision_map(self):
        """
        Creates a simple ASCII map showing player position, direction, terrain and sprites.
        Returns:
            str: A string representation of the ASCII map with legend
        """
        # Get the terrain and movement data
        full_map = self.pyboy.game_wrapper.game_area()
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        downsampled_terrain = self._downsample_array(collision_map)

        # So caves provide an interesting challenge, because there are elevation tiles that are technically passable but
        # can't be reached except by other elevated tiles.
        # Conveniently, elevated tiles are consistently all 261. If any tiles are the rocky edge (279, 305, or 272) it's
        # always either the impassable edge of a platform, or one of those weird semi-hills that can never be walked on.
        # So we can partly "fix" this by detecting 279, 305, or 272 and ruling it out as impassable
        for row in range(9):
            for col in range(10):
                if np.any(full_map[row*2:row*2+2, col*2:col*2+2] == 279) or np.any(full_map[row*2:row*2+2, col*2:col*2+2] == 305) or np.any(full_map[row*2:row*2+2, col*2:col*2+2] == 272):                                                                                                                        
                    downsampled_terrain[row][col] = 0

        # ...however that leaves the pure 261 tiles, which are technically passable but in practice can never be walked into except from another 261 tile or stair tile (or bridges like in victory road)
        # THat can't be handled in the collision map. It has to be dealt with during pathing. Stairs are 277 and 278 btw, but use 22 since that's what the game code does.

        # Get sprite locations
        sprite_locations_and_ids = {x: y for x, y in self.get_sprites()}

        # Get character direction from the full map
        direction = self._get_direction(full_map)
        if direction == "no direction found":
            return None

        # Direction symbols
        direction_chars = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
        player_char = direction_chars.get(direction, "P")

        # Create the ASCII map
        horizontal_border = "+" + "-" * 10 + "+"
        lines = [horizontal_border]

        # Create each row
        for i in range(9):
            row = "|"
            for j in range(10):
                if i == 4 and j == 4:
                    # Player position with direction
                    row += player_char
                elif (j, i) in sprite_locations_and_ids:
                    # Sprite position
                    if sprite_locations_and_ids[(j, i)] in [122, 123]:  # This is pokeball.
                        row += "I"
                    else:
                        row += "S"
                else:
                    # Terrain representation
                    if downsampled_terrain[i][j] == 0:
                        row += "█"  # Wall
                    else:
                        row += "·"  # Path
            row += "|"
            lines.append(row)

        # Add bottom border
        lines.append(horizontal_border)

        # Add legend
        lines.extend(
            [
                "",
                "Legend:",
                "█ - Wall/Obstacle",
                "· - Path/Walkable",
                "S - Sprite",
                "I - Item",
                f"{direction_chars['up']}/{direction_chars['down']}/{direction_chars['left']}/{direction_chars['right']} - Player (facing direction)",
            ]
        )

        # Join all lines with newlines
        return "\n".join(lines)

    def get_valid_moves(self):
        """
        Returns a list of valid moves (up, down, left, right) based on the collision map.
        Returns:
            list[str]: List of valid movement directions
        """
        # Get collision map
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        terrain = self._downsample_array(collision_map)

        # Player is always at position (4,4) in the 9x10 downsampled map
        valid_moves = []

        # We need to check sprites too as they will block traversal. We don't care what the sprites _are_.
        sprites = set(s[0] for s in self.get_sprites())

        # Special casing for warp tiles. If they're at a 0-coordinate we can safely assume the warp transition direction.
        # otherwise I haven't figured out how to figure it out so we just tell the model all directions are valid and just
        # deal with it.
        reader = PokemonRedReader(self.pyboy.memory)
        warp_coords = reader.get_warps()

        # We need absolute coordinates to check warp.
        player_coords = reader.read_coordinates()
        if player_coords in warp_coords:
            if player_coords[0] and player_coords[1]:  # They're both not 9
                return ["up", "down", "left", "right"]  # I have no idea which directions are valid warps so we just fallback on yielding everything. Probably not even worth checking sprites.
            if not player_coords[0]:
                valid_moves.append("left")
            if not player_coords[1]:  # there is a literal corner case where both are 0, but that never happens in Pokemon Red.
                valid_moves.append("up")
        # Check each direction
        if terrain[3][4] != 0 and (4, 3) not in sprites:  # Up
            valid_moves.append("up")
        if terrain[5][4] != 0 and (4, 5) not in sprites:  # Down
            valid_moves.append("down")
        if terrain[4][3] != 0 and (3, 4) not in sprites:  # Left
            valid_moves.append("left")
        if terrain[4][5] != 0 and (5, 4) not in sprites:  # Right
            valid_moves.append("right")

        return valid_moves

    def _can_move_between_tiles(self, tile1: int, tile2: int, tileset: str) -> bool:
        """
        Check if movement between two tiles is allowed based on tile pair collision data.

        Args:
            tile1: The tile being moved from
            tile2: The tile being moved to
            tileset: The current tileset name

        Returns:
            bool: True if movement is allowed, False if blocked
        """
        # Tile pair collision data
        TILE_PAIR_COLLISIONS_LAND = [
            ("CAVERN", 288, 261),
            ("CAVERN", 321, 261),
            ("FOREST", 304, 302),
            ("CAVERN", 298, 261),
            ("CAVERN", 261, 289),
            ("FOREST", 338, 302),
            ("FOREST", 341, 302),
            ("FOREST", 342, 302),
            ("FOREST", 288, 302),
            ("FOREST", 350, 302),
            ("FOREST", 351, 302),
        ]

        TILE_PAIR_COLLISIONS_WATER = [
            ("FOREST", 276, 302),
            ("FOREST", 328, 302),
            ("CAVERN", 276, 261),
        ]

        # Check both land and water collisions
        for ts, t1, t2 in TILE_PAIR_COLLISIONS_LAND + TILE_PAIR_COLLISIONS_WATER:
            if ts == tileset:
                # Check both directions since collisions are bidirectional
                if (tile1 == t1 and tile2 == t2) or (tile1 == t2 and tile2 == t1):
                    return False

        return True

    def get_sprites(self, debug=False) -> set[tuple[tuple[int, int], int]]:
        """
        Get the location of all of the sprites on the screen.
        returns set of coordinates that are (column, row), followed by the bottom sprite tile identifier
        """
        # Group sprites by their exact Y coordinate
        sprites_by_y = {}


        for i in range(40):
            sp = self.pyboy.get_sprite(i)
            if sp.on_screen:
                x = int(sp.x / 160 * 10)
                y = int(sp.y / 144 * 9)
                orig_y = sp.y

                if orig_y not in sprites_by_y:
                    sprites_by_y[orig_y] = []
                sprites_by_y[orig_y].append((x, y, i, sp.tile_identifier))

        # Sort Y coordinates
        y_positions = sorted(sprites_by_y.keys())
        bottom_sprite_tiles: set[tuple[tuple[int, int], int]] = set()

        if debug:
            print("\nSprites grouped by original Y:")
            for orig_y in y_positions:
                sprites = sprites_by_y[orig_y]
                print(f"Y={orig_y}:")
                for x, grid_y, i in sprites:
                    print(f"  Sprite {i}: x={x}, grid_y={grid_y}")

        SPRITE_HEIGHT = 8

        # First, group sprites by X coordinate for each Y level
        for i in range(len(y_positions) - 1):
            y1 = y_positions[i]
            y2 = y_positions[i + 1]

            if y2 - y1 == SPRITE_HEIGHT:
                # Group sprites by X coordinate at each Y level
                sprites_at_y1 = {s[0]: s for s in sprites_by_y[y1]}  # x -> sprite info
                sprites_at_y2 = {s[0]: s for s in sprites_by_y[y2]}

                # Only match sprites that share the same X coordinate
                for x in sprites_at_y2:
                    if x in sprites_at_y1:  # If there's a matching top sprite at this X
                        bottom_sprite = sprites_at_y2[x]
                        # I'm fairly certain bottom_sprite[3] only includes the "bottom" of a top-sprite bottom sprite combo in Red.
                        # But that's okay for our purposes.
                        # If we need top sprite it would just be sprites_at_y1[x][3]
                        bottom_sprite_tiles.add(((x, bottom_sprite[1]), bottom_sprite[3])) 

                        if debug:
                            print(f"\nMatched sprites at x={x}, Y1={y1}, Y2={y2}")

        return bottom_sprite_tiles

    def find_path(self, target_row: int, target_col: int) -> tuple[str, list[str]]:
        """
        Finds the most efficient path from the player's current position (4,4) to the target position.
        If the target is unreachable, finds path to nearest accessible spot.
        Allows ending on a wall tile if that's the target.
        Takes into account terrain, sprite collisions, and tile pair collisions.

        Args:
            target_row: Row index in the 9x10 downsampled map (0-8)
            target_col: Column index in the 9x10 downsampled map (0-9)

        Returns:
            tuple[str, list[str]]: Status message and sequence of movements
        """
        # Get collision map, terrain, and sprites
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        terrain = self._downsample_array(collision_map)
        sprite_locations = set(s[0] for s in self.get_sprites())

        # Get full map for tile values and current tileset
        full_map = self.pyboy.game_wrapper._get_screen_background_tilemap()
        reader = PokemonRedReader(self.pyboy.memory)
        tileset = reader.read_tileset()

        # Start at player position (always 4,4 in the 9x10 grid)
        start = (4, 4)
        end = (target_row, target_col)

        # Validate target position
        if not (0 <= target_row < 9 and 0 <= target_col < 10):
            return "Invalid target coordinates", []

        # A* algorithm
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        # Track closest reachable point
        closest_point = start
        min_distance = heuristic(start, end)

        def reconstruct_path(current):
            path = []
            while current in came_from:
                prev = came_from[current]
                if prev[0] < current[0]:
                    path.append("down")
                elif prev[0] > current[0]:
                    path.append("up")
                elif prev[1] < current[1]:
                    path.append("right")
                else:
                    path.append("left")
                current = prev
            path.reverse()
            return path

        while open_set:
            _, current = heapq.heappop(open_set)

            # Check if we've reached target
            if current == end:
                path = reconstruct_path(current)
                is_wall = terrain[end[0]][end[1]] == 0
                if is_wall:
                    return (
                        f"Partial Success: Your target location is a wall. In case this is intentional, attempting to navigate there.",
                        path,
                    )
                else:
                    return (
                        f"Success: Found path to target at ({target_row}, {target_col}).",
                        path,
                    )

            # Track closest point
            current_distance = heuristic(current, end)
            if current_distance < min_distance:
                closest_point = current
                min_distance = current_distance

            # If we're next to target and target is a wall, we can end here
            if (abs(current[0] - end[0]) + abs(current[1] - end[1])) == 1 and terrain[
                end[0]
            ][end[1]] == 0:
                path = reconstruct_path(current)
                # Add final move onto wall
                if end[0] > current[0]:
                    path.append("down")
                elif end[0] < current[0]:
                    path.append("up")
                elif end[1] > current[1]:
                    path.append("right")
                else:
                    path.append("left")
                return (
                    f"Success: Found path to position adjacent to wall at ({target_row}, {target_col}).",
                    path,
                )

            # Check all four directions
            for dr, dc, direction in [
                (1, 0, "down"),
                (-1, 0, "up"),
                (0, 1, "right"),
                (0, -1, "left"),
            ]:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds
                if not (0 <= neighbor[0] < 9 and 0 <= neighbor[1] < 10):
                    continue
                # Skip walls unless it's the final destination
                if terrain[neighbor[0]][neighbor[1]] == 0 and neighbor != end:
                    continue
                # Skip sprites unless it's the final destination
                if (neighbor[1], neighbor[0]) in sprite_locations and neighbor != end:
                    continue

                # Check tile pair collisions
                # Get bottom-left tile of each 2x2 block
                current_tile = full_map[current[0] * 2 + 1][
                    current[1] * 2
                ]  # Bottom-left tile of current block
                neighbor_tile = full_map[neighbor[0] * 2 + 1][
                    neighbor[1] * 2
                ]  # Bottom-left tile of neighbor block
                if not self._can_move_between_tiles(
                    current_tile, neighbor_tile, tileset
                ):
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # If target unreachable, return path to closest point
        if closest_point != start:
            path = reconstruct_path(closest_point)
            return (
                f"Partial Success: Could not reach the exact target, but found a path to the closest reachable point.",
                path,
            )

        return (
            "Failure: No path is visible to the chosen location. You may need to explore a totally different path to get where you're trying to go.",
            [],
        )
    
    def get_in_combat(self) -> bool:
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_in_combat()
    
    def get_warps(self) -> list[tuple[int, int]]:
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.get_warps()
    
    def get_direction(self) -> str:
        full_map = self.pyboy.game_wrapper.game_area()
        return self._get_direction(full_map)

    def get_state_from_memory(self, get_nearby_warps: bool=True, display_warp_functor: Optional[Callable[[int, int], bool]] = None) -> tuple[str, str, tuple[int, int]]:
        """
        Reads the game state from memory and returns a string representation of it.

        # display_warp_functor is a function that, given a warp coordinate, returns whether it should be showed to the model.
        """
        reader = PokemonRedReader(self.pyboy.memory)
        memory_str = ""

        name = reader.read_player_name()
        if name == "NINTEN":
            name = "Not yet set"
        rival_name = reader.read_rival_name()
        if rival_name == "SONY":
            rival_name = "Not yet set"

        # Get valid moves
        valid_moves = self.get_valid_moves()
        valid_moves_str = ", ".join(valid_moves) if valid_moves else "None"

        location = reader.read_location()
        coords = reader.read_coordinates()  # This comes out col, row

        full_map = self.pyboy.game_wrapper.game_area()
        direction = self._get_direction(full_map)

        memory_str += f"Player: {name}\n"
        memory_str += f"Character is currently facing: {direction}\n"
        memory_str += f"Rival: {rival_name}\n"
        memory_str += f"Money: ${reader.read_money()}\n"
        memory_str += f"RAM Location: {location}\n"
        memory_str += f"Coordinates (Horizontal Position/column left-to-right, Vertical Position/row top-to-bottom): {coords}\n"
        memory_str += f"Valid Moves: {valid_moves_str}\n"
        if get_nearby_warps:
            all_warps = reader.get_warps()
            nearby_warps = []
            for entry in all_warps:
                if display_warp_functor is not None:
                    if display_warp_functor(entry[0], entry[1]):
                        nearby_warps.append(entry)
                elif (entry[0] - coords[0] < 6 or coords[0] - entry[0] < 5) and abs(entry[1] - coords[1]) < 5:
                    nearby_warps.append(entry)
            memory_str += f"Warps (Doors, stairs, etc.) in revealed text map: {nearby_warps}\n"
        memory_str += f"Badges: {', '.join(reader.read_badges())}\n"

        # Inventory
        memory_str += "Inventory:\n"
        for item, qty in reader.read_items():
            memory_str += f"  {item} x{qty}\n"

        # Dialog
        dialog = self.get_stored_dialogue()
        if dialog:
            memory_str += f"Dialog: {dialog}\n"
        else:
            memory_str += "Dialog: None\n"

        # Party Pokemon
        memory_str += "\nPokemon Party:\n"
        for pokemon in reader.read_party_pokemon():
            memory_str += f"\n{pokemon.nickname} ({pokemon.species_name}):\n"
            memory_str += f"Level {pokemon.level} - HP: {pokemon.current_hp}/{pokemon.max_hp}\n"
            memory_str += f"Types: {pokemon.type1.name}{', ' + pokemon.type2.name if pokemon.type2 else ''}\n"
            for move, pp in zip(pokemon.moves, pokemon.move_pp, strict=True):
                memory_str += f"- {move} (PP: {pp})\n"
            if pokemon.status != StatusCondition.NONE:
                memory_str += f"Status: {pokemon.status.get_status_name()}\n"
            else:
                memory_str += f"Status: Normal\n"

        return memory_str, location, coords

    def stop(self):
        with self.pyboy_lock(1):
            self.button_queue_clear.clear()
            self.button_queue.put(("stop", None))
        self.is_dialogue_running = False