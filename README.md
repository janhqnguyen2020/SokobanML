# Comparing Classical Search and Reinforcement Learning Methods for Solving Sokoban

## Project Summary

This project compares classical search algorithms and reinforcement learning methods for solving Sokoban puzzles. In Sokoban, the player must push all boxes onto designated goal tiles. Boxes can only be pushed, so an incorrect move may create an irreversible deadlock and make the puzzle unsolvable.

We implement Breadth-First Search (BFS), Greedy Best-First Search, A* Search, and a Deep Q-Network (DQN). The methods are evaluated based on success rate, solution length, runtime, search efficiency, and generalization to unseen maps.

## Approaches

### Classical Planning

The classical planners represent each Sokoban state using the player position and the set of box positions. Successor states correspond to valid box pushes rather than individual player movements.

Before search begins, dead squares are identified. Pushes that place boxes on dead squares or create known deadlock configurations are removed from the search. Each planner uses a maximum of 200,000 expanded nodes.

#### Breadth-First Search

BFS expands states in increasing push depth. It can find a shortest-push solution within the expansion limit, but its runtime and memory usage increase quickly on difficult maps.

#### Greedy Best-First Search

Greedy Best-First Search prioritizes states using the estimated distance between boxes and goals. It is often faster than BFS but may make short-sighted decisions because it does not consider the number of pushes already taken.

#### A* Search

A* combines the number of pushes already taken with an estimate of the remaining cost. The heuristic uses Hungarian matching to find the minimum-distance assignment between boxes and goals. This provides a balance between solution quality and search efficiency.

### Reinforcement Learning

#### Primitive-Action DQN

The initial DQN selected low-level movement actions: up, down, left, and right. This approach performed poorly because useful rewards were separated by long navigation sequences. The agent rarely discovered successful box-pushing behavior within a practical training budget.

#### High-Level DQN

The final DQN selects macro-actions consisting of a box and a push direction. The environment handles the player movement required to perform the selected push.

The high-level environment includes:

- **Action masking:** Prevents pushes into walls, occupied cells, dead squares, or unreachable push positions.
- **Reward shaping:** Rewards box-goal progress and penalizes repeated states, reversed pushes, deadlocks, and invalid actions.
- **CNN state encoding:** Represents walls, goals, boxes, and the player as four binary board layers.
- **Double DQN updates:** Uses separate online and target networks to reduce value overestimation.
- **Curriculum training:** Gradually shifts training from one-box puzzles toward more difficult three-box puzzles.
- **Imitation learning:** Uses A* demonstrations for behavior-cloning pretraining and continued expert guidance during reinforcement learning.

Training used 340 fixed maps together with procedurally generated Sokoban environments.

## Evaluation

The methods were evaluated on two benchmarks.

### FinalEval Fixed Benchmark

A fixed collection of 150 unseen maps:

- 50 one-box maps
- 50 two-box maps
- 50 three-box maps

This benchmark provides a reproducible comparison across all methods.

### Procedural Generalization Benchmark

A collection of 100 `Sokoban-small-v1` episodes generated using fixed evaluation seeds. This benchmark measures performance on unseen layouts containing interior walls and constrained movement.

## Results

| Method | FinalEval Success | Procedural Success | Main Observation |
|---|---:|---:|---|
| BFS | 100% | 100% | Reliable but computationally expensive |
| Greedy Best-First | 100% | 100% | Fast search but not guaranteed to be optimal |
| A* | 100% | 100% | Strong balance of solution quality and efficiency |
| High-Level DQN | 64% | 51% | Fast inference but weaker long-horizon planning |

The DQN solved all one-box maps and 82% of two-box maps in FinalEval, but solved only 10% of three-box maps. Most failures were caused by repeated-state loops or irreversible dead-end configurations rather than invalid actions.

Overall, A* was the most effective method at the tested puzzle sizes. The DQN learned reusable box-pushing behavior and provided fast inference after training, but it remained less reliable on complex multi-box puzzles.

## Installation

Install the required packages from the project root directory:

```bash
pip install -r requirements.txt
```

## How to Run

Run all commands from the project root directory.

### Classical Planners

Run all three planners:

```bash
python run_experiments.py
```

Run the planners on the curriculum maps:

```bash
python run_experiments.py --sources curriculum --algorithms all
```

Run the procedural `Sokoban-small-v1` benchmark:

```bash
python run_experiments.py --sources original --envs small_v1 --algorithms all --episodes 100
```

#### Planner Options

| Flag | Accepted Values | Description |
|---|---|---|
| `--sources` | `all`, `custom_core`, `canvas`, `curriculum`, `additional`, `archived`, `final_eval`, `original` | Selects the map or environment source |
| `--algorithms` | `all`, `bfs`, `greedy`, `astar` | Selects which planners to run |
| `--envs` | `v0`, `v1`, `v2`, `small_v0`, `small_v1`, `large_v0`, `large_v1`, `large_v2`, `huge_v0` | Selects procedural environments when using `--sources original` |
| `--maps` | One or more map names | Runs only the selected fixed maps |
| `--episodes` | Integer | Sets the number of procedural episodes |
| `--seed-base` | Integer | Sets the first evaluation seed |
| `--show-ui` | No value | Displays the puzzle during evaluation |
| `--delay` | Decimal number | Sets the delay between displayed steps |

Example using only A* with visualization:

```bash
python run_experiments.py --sources curriculum --algorithms astar --show-ui --delay 0.2
```

Planner results are saved as CSV files under `results/`.

### Train the High-Level DQN

Train the DQN with curriculum learning:

```bash
python main_curriculum_dqn.py
```

Warm-start training from an existing model:

```bash
python main_curriculum_dqn.py --init-model <path-to-model.zip>
```

### Train with Imitation Learning

Train the curriculum DQN using A* demonstrations and behavior cloning:

```bash
python main_curriculum_dqn_with_imitation.py
```

Common imitation-training options include:

| Flag | Description |
|---|---|
| `--imitation-epochs` | Sets the number of behavior-cloning epochs |
| `--imitation-batch-size` | Sets the behavior-cloning batch size |
| `--imitation-learning-rate` | Sets the behavior-cloning learning rate |
| `--max-demonstrations` | Limits the number of expert demonstrations |
| `--regenerate-demos` | Regenerates the A* demonstration dataset |
| `--skip-imitation` | Skips behavior-cloning pretraining |
| `--resume-model` | Resumes from an existing model checkpoint |
| `--resume-replay-buffer` | Loads an existing replay buffer |
| `--start-phase` | Selects the first curriculum phase |
| `--stop-after-phase` | Stops after the selected curriculum phase |

### Evaluate a Trained DQN

Evaluate the latest model on the 150-map FinalEval benchmark:

```bash
python eval_dqn.py --mode finalEval --boxes all --episodes 1
```

Evaluate the latest model on 100 procedural episodes:

```bash
python eval_dqn.py --mode procedural --episodes 100
```

Evaluate a specific model:

```bash
python eval_dqn.py --model <path-to-model.zip> --mode finalEval
```

#### DQN Evaluation Options

| Flag | Accepted Values | Description |
|---|---|---|
| `--mode` | `procedural`, `currTrain`, `finalEval` | Selects the evaluation benchmark |
| `--boxes` | `1`, `2`, `3`, `all` | Filters fixed maps by box count |
| `--episodes` | Integer | Sets the number of episodes per map or procedural episodes |
| `--model` | Path to a `.zip` file | Selects a model checkpoint instead of the latest model |
| `--maps` | One or more map names | Evaluates only the selected fixed maps |
| `--show-ui` | No value | Displays the puzzle during evaluation |
| `--save` | No value | Saves evaluation results and videos |
| `--note` | Text label | Adds a label to the saved evaluation |

Example:

```bash
python eval_dqn.py \
  --model <path-to-model.zip> \
  --mode finalEval \
  --boxes 3 \
  --episodes 1 \
  --save \
  --note three_box_test
```

### View All Options

Use `--help` to view the complete command-line options for a script:

```bash
python run_experiments.py --help
python eval_dqn.py --help
python main_curriculum_dqn.py --help
python main_curriculum_dqn_with_imitation.py --help
```

## Team Members

- Quang Dinh Tue Tran
- Shizuka Takao
- Joseph Anh-Quoc Nguyen
