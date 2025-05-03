import argparse
import logging
import os
import time
import sys 
from typing import Literal
from agent.simple_agent import SimpleAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Claude Plays Pokemon - Starter Version")
    parser.add_argument(
        "--rom", 
        type=str, 
        default="pokemon.gb",
        help="Path to the Pokemon ROM file"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=10, 
        help="Number of agent steps to run"
    )
    parser.add_argument(
        "--display", 
        action="store_true", 
        help="Run with display (not headless)"
    )
    parser.add_argument(
        "--sound", 
        action="store_true", 
        help="Enable sound (only applicable with display)"
    )
    parser.add_argument(
        "--max-history", 
        type=int, 
        default=50, 
        help="Maximum number of messages in history before summarization"
    )
    parser.add_argument(
        "--load-state", 
        type=str, 
        default=None, 
        help="Path to a saved state to load"
    )
    parser.add_argument(
        "--main-thread-target", 
        type=str, 
        choices=["emulator", "agent", "auto"],
        default="auto",
        help="Run pyboy in the main thread"
    )

    args = parser.parse_args()
    
    # Get absolute path to ROM
    if not os.path.isabs(args.rom):
        rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.rom)
    else:
        rom_path = args.rom
    
    # Check if ROM exists
    if not os.path.exists(rom_path):
        logger.error(f"ROM file not found: {rom_path}")
        print("\nYou need to provide a Pokemon Red ROM file to run this program.")
        print("Place the ROM in the root directory or specify its path with --rom.")
        return
    from pyboy import logging
    logging.log_level("DEBUG")

    # default to running pyboy in the main thread on macOS
    pyboy_main_thread = sys.platform == "darwin"

    if args.main_thread_target == "emulator":
        pyboy_main_thread = True
    elif args.main_thread_target == "agent":
        pyboy_main_thread = False

    # Create and run agent
    agent = SimpleAgent(
        rom_path=rom_path,
        headless=not args.display,
        sound=args.sound if args.display else False,
        max_history=args.max_history,
        load_state=args.load_state,
        location_archive_file_name="locations.pkl",
        pyboy_main_thread=pyboy_main_thread
    )

    try:
        logger.info(f"Starting agent for {args.steps} steps")
        steps_completed = agent.run(num_steps=args.steps, save_file_name="save.state")
        logger.info(f"Agent completed {steps_completed} steps")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping")
    except AssertionError as e:
        logger.error(f"Error running agent: {e}")
    finally:
        agent.stop()

if __name__ == "__main__":
    main()