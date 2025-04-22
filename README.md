# Claude Plays Pokemon - Elaborate Version

As seen on: https://www.lesswrong.com/posts/8aPyKyRrMAQatFSnG

An of Claude playing Pokemon Red using the PyBoy emulator, based (but heavily elaborated on) on the original starter code provided by David Hershey of Anthropic (https://github.com/davidhershey/ClaudePlaysPokemonStarter/tree/main) 

The starter version included:

- Simple agent that uses Claude to play Pokemon Red
- Memory reading functionality to extract game state information
- Basic emulator control through Claude's function calling

This repo adds:

- Support for o3/o4-mini and Gemini-2.5 (Note: May have been broken recently with recent changes, sorry, but should be easy to fix)
- Much more elaborate scaffold features:

1. A running ASCII collision map of each location is made as Claude explores and is provided to the models
2. Logging every time Claude enters a new location for the first time
3. An overlay inspired by both Claude and GeminiPlaysPokemon
4. Labels automatically recorded on the map and in text when a location changes
5. The model can "mark_checkpoint" to record achievements and keep track of progress.
6. Claude can mark labels on the map as desired, which are recorded for the future
7. A new more elaborate 3-stage "Meta-Critique Claude" that shows up at context summary and tries to keep an organized accounting of game state and facts
8. A Navigation Assist tool, which is just an instance of the model instructed to study the ASCII map carefully and look for where to go. (Not fully tested, unsure how much it helps)

Features NOT included (that you may be familiar with from e.g. ClaudePlaysPokemon):

1. Separately threaded emulator (So, the emulator will pause every time the model is thinking, rather than continuing to run and play music etc.)
2. A memory file management system a la ClaudePlaysPokemon

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Add your API keys to a "secret_api_keys.py" file in the top level.

4. Place your Pokemon Red Color Hack ROM file in the root directory (you need to provide your own ROM)

## Usage

Run the main script:

```
python main.py
```

Optional arguments:
- `--rom`: Path to the Pokemon ROM file (default: `pokemon.gb` in the root directory)
- `--steps`: Number of agent steps to run (default: 10)
- `--display`: Run with display (not headless)
- `--sound`: Enable sound (only applicable with display)
- `--load-state`: Load a previous save state (currently goes to "save.state"). 
                  NOTE: this is only the emulator state, the scaffold state is hard-coded to "locations.pkl". Sorry >_> A full save state is thus "save.state" and "locations.pkl"

Example:
```
python main.py --rom pokemon.gb --steps 20 --display --sound
python main.py --rom pokemon.gb --steps 20 --display --sound --load-state ./save.state
```

Note: You may keyboard interrupt the bot at any time and it will *usually* automatically save, but if you catch the emulator mid-save-state write or something it may corrupt the save files. Keep backups! I've generally found that if you interrupt and it goes to a breakpoint it's a good time to go manually copy your save files.

## Implementation Details

### Components

- `agent/simple_agent.py`: Main agent class that uses Claude to play Pokemon
- `agent/emulator.py`: Wrapper around PyBoy with helper functions
- `agent/memory_reader.py`: Extracts game state information from emulator memory
- `agent/prompts.py`: System prompts for the various internal agents etc.
- `agent/tool_definitions.py`: Tool Definitions for the tools the models can call.

### How It Works

1. The agent captures a screenshot from the emulator
2. It reads the game state information from memory
3. It sends the screenshot and game state to the model
4. The model responds with explanations and emulator commands
5. The agent executes the commands and repeats the process
6. Every once in a while when max_messages is reached, a summary call is made that also tries to do fact sorting and critiquing.
