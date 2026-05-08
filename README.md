# [Cottonmouth](https://github.com/einsteintoolkit/Cottonmouth)

<img src="https://cct.lsu.edu/~lsanches/cottonmouth.png" width="400" align="right" />

**Cottonmouth** is a suite of astrophysics thorns for [Cactus](https://cactuscode.org/) with the CarpetX driver. Cottonmouth was generated using [the Einstein Engine](https://github.com/max-morris/EinsteinEngine.git).

## Overview

Cottonmouth is ready for production. It currently contains the following thorns:
- **CottonmouthBSSNOK4m**: Evolves the BSSNOK formulation of Einstein's equations with **4**th order stencils and **m**atter terms enabled.
- **CottonmouthZ4c4m**: Evolves the Z4c formulation of Einstein's equations with **4**th order stencils and **m**atter terms enabled.
- **CottonmouthGaugeWaveID**: Provides gauge wave initial data according to [this formulation](https://arxiv.org/abs/0709.3559).
- **CottonmouthLinearWaveID**: Provides linear wave initial data according to [this formulation](https://arxiv.org/abs/0709.3559).

## Getting started

You might want to try take a look at the test parameter file (arrangements/Cottonmouth/CottonmouthBSSNOK/test/qc0.par) to see what it can do.

## Customization
Cottonmouth was authored with the [the Einstein Engine](https://github.com/max-morris/EinsteinEngine.git). Inside the Einstein Engine repo, you can find the [Cottonmouth recipe group](https://github.com/max-morris/EinsteinEngine/tree/master/recipes/Cottonmouth), which is the "source code" from which Cottonmouth was generated. If you want to customize the stencil order or whether matter terms are enabled, you can use the Einstein Engine to generate your own copy of the thorn, passing these options as generation-time flags. In a future release, we will add the capability to set these parameters at runtime. You can make more advanced customizations by tweaking the recipes yourself.

## Licensing and Attribution
Cottonmouth was authored by Lucas Timotheo Sanches, Max Morris, and Steven R. Brandt.
It is licensed under version 3 of the Affero General Public License (AGPLv3).

