# LLM Pokémon Scaffold - Elaborate Version

![LLM Pokémon Scaffold](https://res.cloudinary.com/lesswrong-2-0/image/upload/c_scale,w_250/f_auto,q_auto/v1/mirroredImages/8aPyKyRrMAQatFSnG/fcqugqcpuloqkloqz9bw)

An agent harness that helps LLMs play Pokemon Red using the PyBoy emulator, forked from the [original starter code](<https://github.com/davidhershey/ClaudePlaysPokemonStarter/tree/main>) provided by David Hershey of Anthropic.

As seen on: [Research Notes: Running Claude 3.7, Gemini 2.5 Pro, and o3 on Pokémon Red](https://www.lesswrong.com/posts/8aPyKyRrMAQatFSnG).

## Explanation

This is NO LONGER a basic scaffold. In fact, it adds quite a lot to try to help LLMs perform, partly see just what is necessary.

The starter version included:

- Simple agent that uses an LLM to play Pokemon Red
- Memory reading functionality to extract game state information
- Basic emulator control through the LLM's function calling

This repo adds support for o3/o4-mini and Gemini 2.5 Pro/Flash,
along with much more elaborate features.

New features include (works for all LLMs):

### Reasoning Aids

1. A new more elaborate 3-stage "Meta-Critique LLM" that shows up at context summary and tries to keep an organized accounting of game state and facts -- this greatly helps keep the model on track, but uses its inherent knowledge of Red
2. A checkpoint log to keep a history of events (and prevent hallucinations.)
   - The model uses "mark_checkpoint" to record these

### Navigation Aids

1. A running ASCII collision map of each location is made as the LLM explores and is provided to the models  
   1a. Very handholdy -- it also now gives numbers indicating how far away various tiles are to reach
2. An overlay inspired by both Claude and GeminiPlaysPokemon
3. Labels automatically recorded on the map and in text when a location changes. The model can also add labels as desired
4. A Detailed Navigator tool, which is just an instance of the model instructed to study the ASCII map carefully and look for where to go, without being told where (this helps in mazes)
5. A new tool that will auto-path the LLM to a location in an area that it knows the coordinates of  
   10a. This could have been done instructing the LLM to verbally run the algorithm, but is very token-expensive and slow. You can switch it back to the LLM in config (note: you'll have to jack up the tokens in the code and implement streaming for it work for great distances)
6. Numerous small scaffold improvements (like not lying to the model about its available moves at the edge of warp boundaries or next to Sprites)

### Quality of Life

1. Logging every time the LLM enters a new location for the first time, for tracking progress
2. Separate emulator threading, so it will keep running while the agent is thinking rather than pausing

### Features NOT included (that you may be familiar with from e.g. ClaudePlaysPokemon)

1. A memory file management system

## Setup

Recommended Python 3.11. That's how this was written and I think >3.11 breaks the current versions of Pyboy.

1. Clone this repository
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your API keys to a "secret_api_keys.py" file in the top level.

4. Place your Pokemon Red Color Hack ROM file in the root directory (you need to provide your own ROM)

## Usage

Check config.py for key model configuration settings you may want to change (like what model to use)

Run the main script:

```bash
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

```bash
python main.py --rom pokemon.gb --steps 20 --display --sound
python main.py --rom pokemon.gb --steps 20 --display --sound --load-state ./save.state
```

Note: You may keyboard interrupt the bot at any time and it will *usually* automatically save, but if you catch the emulator mid-save-state write or something it may corrupt the save files. Keep backups! I've generally found that if you interrupt and it goes to a breakpoint it's a good time to go manually copy your save files.

## Implementation Details

### Components

- `agent/simple_agent.py`: Main agent class that uses LLM to play Pokemon
- `agent/emulator.py`: Wrapper around PyBoy with helper functions
- `agent/memory_reader.py`: Extracts game state information from emulator memory
- `agent/prompts.py`: System prompts for the various internal agents
- `agent/tool_definitions.py`: Defines the tools models can call
- `agent/utils.py`: Various utilities, currently just Gemini stuff

### How It Works

1. The agent captures a screenshot from the emulator
2. It reads the game state information from memory
3. It sends the screenshot and game state to the model
4. The model responds with explanations and emulator commands
5. The agent executes the commands and repeats the process
6. Every once in a while when max_messages is reached, a summary call is made that also tries to do fact sorting and critiquing
