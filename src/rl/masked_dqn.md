# Masked DQN Guide

This file explains [masked_dqn.py](C:/Users/pc/Desktop/projects/SokobanStuff/SokobanML-experiment/src/rl/masked_dqn.py) at a high level.

The short version is:

- normal DQN assumes every action is always available
- your high-level Sokoban env does **not** work like that
- some macro actions are valid and some are invalid
- `MaskedDQN` makes DQN respect that valid-action mask

## The Main Problem

In your high-level Sokoban setup, the action space is fixed in size.

Example:

- `3` boxes
- `4` directions each
- total macro actions = `12`

But on any specific board state:

- some pushes are possible
- some pushes are impossible
- some action ids should not be chosen at all

A normal DQN would still treat all `12` actions like they are equally available.

That causes problems:

- random exploration wastes time on impossible moves
- greedy action selection may prefer invalid moves
- target Q-values may bootstrap through invalid moves

`MaskedDQN` fixes that.

## Where The Mask Comes From

The high-level environment stores the valid-action mask at the **end of the observation vector**.

So one observation looks like:

1. board layers
2. optional scalar features
3. action mask

The action mask is a binary tail like:

```python
[1, 0, 1, 1, 0, 0, ...]
```

Meaning:

- `1` = this macro action is valid
- `0` = this macro action is invalid

`MaskedDQN` reads that tail and uses it when choosing and training actions.

## What `MaskedDQN` Changes

This class changes three important parts of normal DQN:

1. exploration
2. greedy action choice
3. target value calculation during training

That is the whole point of the file.

## Function By Function

### `__init__(...)`

This sets up the masked DQN.

Important job:

- decide how wide the action mask is

Usually that width is just:

- `action_space.n`

So if the env has `12` macro actions, the mask size is `12`.

## `_extract_action_masks(...)`

This reads the mask from the observation tail.

Simple meaning:

- take the observation
- slice off the last `action_mask_size` values
- return those values as the mask

This is how the model knows which actions are currently legal.

## `_masked_q_values(...)`

This is one of the most important functions.

It takes:

- raw Q-values from the network
- the observation tensor

Then it:

- finds which actions are valid from the mask
- sets invalid Q-values to a huge negative number like `-1e9`

Why?

Because then:

```python
argmax(masked_q_values)
```

will ignore invalid actions.

So this function is basically:

- "hide the illegal actions before choosing the best action"

## `_sample_masked_random_actions(...)`

This handles random exploration.

Instead of sampling from all actions blindly, it:

- looks at the valid actions
- samples uniformly from only the valid ones

If a row somehow has no valid actions:

- it falls back to sampling from all actions

Simple meaning:

- "when acting randomly, be random only over legal moves"

## `_masked_greedy_actions(...)`

This handles greedy action choice.

It:

1. runs the Q-network
2. masks invalid Q-values
3. picks the best remaining action

Simple meaning:

- "when acting greedily, choose the best legal move"

## `predict(...)`

This matches the usual Stable-Baselines3 `predict()` API.

It decides between:

- random masked exploration
- masked greedy action selection

So this is the main action-selection function used outside training too.

Simple meaning:

- "give me an action, but never ignore the mask"

## `_sample_action(...)`

This is used during rollout collection in SB3.

It changes behavior depending on training stage:

- before learning starts: sample masked random actions
- after learning starts: use `predict(...)`

Simple meaning:

- "during data collection, still respect valid actions"

## `train(...)`

This is the training update loop.

This is where masking also affects the Bellman target, not just acting.

For each gradient step:

1. sample replay data
2. run online network on next observations
3. mask invalid next-state actions
4. choose best valid next action
5. run target network on next observations
6. mask invalid next-state target Q-values
7. gather the target value of the chosen valid next action
8. build DQN targets
9. compute loss
10. do optimizer step

Why this matters:

- if training ignored the mask here, the target could learn from impossible actions

So this function makes training consistent with the env rules.

## The Big Idea In One Sentence

Normal DQN says:

- "pick the action with the biggest Q-value"

`MaskedDQN` says:

- "pick the action with the biggest Q-value **among the legal actions only**"

That is the whole idea.

## Why This File Matters In Your Project

Your high-level Sokoban agent does not move in raw primitive steps.

It chooses macro pushes.

That means the environment naturally has many invalid choices at each state.

Without `MaskedDQN`, the model would:

- explore lots of nonsense moves
- learn weaker targets
- waste training signal

So this file is a key part of making the high-level DQN setup workable.

## Short Summary

If you want the shortest mental model:

- env creates valid-action mask
- mask is stored at end of observation
- `MaskedDQN` reads that mask
- invalid actions get hidden during:
  - random action choice
  - greedy action choice
  - target Q-value calculation

So `masked_dqn.py` is basically:

- **DQN, but aware that some actions are illegal in each state**
