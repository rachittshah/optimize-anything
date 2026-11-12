# Figures

The images used in the README and RESULTS.md. Each comes in a light and a dark version; GitHub picks one to match the reader's theme.

| File | What it shows | Built by |
|---|---|---|
| `loop-{light,dark}.svg` | One round of optimization, drawn as a small bench machine (animated) | `build/loop-rig.mjs` |
| `results-engine-*.svg` | Original vs new engine on held-out test, and cost | `build/charts.mjs` |
| `results-upgrades-*.svg` | Every upgrade tried, with 95% intervals | `build/charts.mjs` |
| `results-skill-*.svg` | The skill before and after optimization | `build/charts.mjs` |

Rebuild:

```bash
node docs/figures/build/charts.mjs                                   # no dependencies
ANATOMY_KIT=/path/to/anatomy/kit node docs/figures/build/loop-rig.mjs   # needs the "anatomy" isometric drawing kit
```

The chart numbers are the ones in RESULTS.md. If results change, update the data at the top of each chart function in `build/charts.mjs`.
