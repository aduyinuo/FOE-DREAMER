# FOE-Dreamer

Anonymous accompanying repository for the submitted manuscript
"FOE-Dreamer: Deployment-Efficient Learning of Cyber Defense Policies
in Operational Networks."

This repository contains the modeling code, the cyber-defense environment
used for the on-testbed evaluation, and the training and evaluation
drivers.

## Layout

```
foe_dreamer/
  models/
    world_model.py          # RSSM (recurrent state-space model)
    opponent_model.py       # Foe (opponent) heads
    heads.py                # Encoder/decoder/actor/critic/reward/discount
    losses.py               # ELBO, KL balancing, foe-prediction, AC loss
  envs/
    cyber_env.py            # Cyber-defense environment
    c2_bridge.py            # gRPC client for the controller-side C2 server
    scenarios/              # YAML scenario specs (small, medium)
    playbooks/              # Ansible playbooks: deploy / reset / teardown
  interaction/
    replay.py               # Episodic replay buffer
    rollout.py              # Episode collection + greedy eval
    train.py                # Training driver
    eval.py                 # Greedy evaluation from checkpoint
  configs/
    foe_dreamer.yaml        # Default config
  requirements.txt
```

## Quick start

```
pip install -r requirements.txt
python -m interaction.train --config configs/foe_dreamer.yaml
```

## Mapping to the paper

| Paper section                           | Code path                                         |
|-----------------------------------------|---------------------------------------------------|
| §5.2 Factored World Model                 | `models/world_model.py`                           |
| §5.3 Opponent Model                       | `models/opponent_model.py`                        |
| §5.4 Live Training Loop (imagination + actor-critic)         | `models/losses.py::actor_critic_loss`             |
| §4.3 Testbed                          | `envs/cyber_env.py`, `envs/playbooks/`         |
| §5.4 Training procedure                 | `interaction/train.py`                            |
| §6.3 Foe-head ablation                  | set `train.beta_opponent: 0.0` in the config      |

## Notice

References to prior work appear only as citations in the manuscript.
