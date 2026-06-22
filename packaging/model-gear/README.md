# model-gear (deprecated alias)

`model-gear` has been renamed to **`lobes-cli`** (command: `lobes`).

This package is a thin alias: installing it pulls in `lobes-cli`. Please install
`lobes-cli` directly going forward:

```bash
pip install lobes-cli      # provides the `lobes` command (and the legacy `model` alias)
```

The `model-gear` distribution will continue to track `lobes-cli` for one or more
releases to avoid breaking existing installs.
