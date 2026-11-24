# Research attribution

ModeExplorer independently implements Krea-specific adaptations of concepts attributed in the [research notes](docs/research.md). It vendors no reference implementation, ComfyUI source, model weights, encoder, or VAE. Those components remain subject to their respective licenses.

## ReDiverse

The ReDiverse integration (`mode_explorer/rediverse.py`, `mode_explorer/rediverse_runtime.py`, `mode_explorer/covariance.py`) follows the method, reference constants, seed rules, direction-estimation formula and adapter key layout of [ReDiverse](https://github.com/H-furkan/rediverse) (Horoz, Kara, Oktay, Akan, Kara, Bhattad and Rehg; revision `0a39a7ce38c783038e2394c72d1b877c0129552b`), licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). The ModeExplorer implementation was written for native ComfyUI and contains no copied upstream source files; the adapted formulas and constants are attributed here. ReDiverse is distributed on an "AS IS" basis, without warranties or conditions of any kind.

The released Krea 2 ReDiverse adapter weights are a Derivative of Krea 2, governed by the Krea 2 Community License Agreement and Krea's Acceptable Use Policy. They are not included in this repository or its releases.
