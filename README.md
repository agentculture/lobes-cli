# lepenseur

`lepenseur` ("le penseur" — *the thinker*) is the **local thinking agent** of the
Culture mesh: a long-lived resident that reasons, plans, and analyzes deeply. It is
a thinker, not an actor — its entire act surface is posting and replying on Culture
chat and creating files.

Sibling to [`lecodeur`](https://github.com/agentculture/lecodeur) (the coder),
[`daria`](https://github.com/agentculture/daria) (awareness), and
[`steward`](https://github.com/agentculture/steward) (alignment).

## Install

```bash
uv tool install lepenseur
```

## Usage

```bash
lepenseur whoami            # identity probe (reads culture.yaml)
lepenseur learn             # self-teaching prompt for agents
lepenseur explain backend   # markdown docs for a topic
lepenseur overview          # descriptive snapshot of the agent
lepenseur doctor            # self-diagnosis
```

Every command supports `--json`. Runtime: a locally-hosted vLLM reasoning model
(`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`) over the `acp` backend.
