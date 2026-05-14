# [Cottonmouth](https://github.com/einsteintoolkit/Cottonmouth)

<img src="https://cct.lsu.edu/~lsanches/cottonmouth.png" width="400" align="right" />

**Cottonmouth** is a suite of astrophysics thorns for [Cactus](https://cactuscode.org/) with the CarpetX driver. Cottonmouth was generated using [the Einstein Engine](https://github.com/max-morris/EinsteinEngine.git).

## Overview

Cottonmouth is ready for production. It currently contains the following thorns:
- **CottonmouthBSSNOK4m**: Evolves the BSSNOK formulation of Einstein's equations with **4**th order stencils and **m**atter terms enabled.
- **CottonmouthZ4c4m**: Evolves the Z4c formulation of Einstein's equations with **4**th order stencils and **m**atter terms enabled.
- **CottonmouthGaugeWaveID**: Provides gauge wave initial data according to [this formulation](https://arxiv.org/abs/0709.3559).
- **CottonmouthLinearWaveID**: Provides linear wave initial data according to [this formulation](https://arxiv.org/abs/0709.3559).

Both CottonmouthBSSNOK and CottonmouthZ4c use the *puncture gauge*, with $1+\log$ lapse and the $\Gamma$-driver shift.

## Customization
Cottonmouth was authored with the [the Einstein Engine](https://github.com/max-morris/EinsteinEngine.git). Inside the Einstein Engine repo, you can find the [Cottonmouth recipe group](https://github.com/max-morris/EinsteinEngine/tree/master/recipes/Cottonmouth), which is the "source code" from which Cottonmouth was generated. If you want to customize the stencil order or whether matter terms are enabled, you can use the Einstein Engine to generate your own copy of the thorn, passing these options as generation-time flags. In a future release, we will add the capability to set these parameters at runtime. You can make more advanced customizations by tweaking the recipes yourself.

## Getting started

The best way to get started with your own simulations using Cottonmouth is to look at our [example parameter files](https://github.com/max-morris/EinsteinEngine/tree/master/recipes/Cottonmouth/parfiles), which currently include:
1. Short BBH collision simulations in the `qc0` configuration, using both the BSSNOK and Z4c formulations.
2. Magnetized TOV star simulation using `AsterX` as the GRMHD driver.

These parameter files should be readily adaptable into more interesting simulations; they include features such as multiple refinement levels with puncture tracking, apparent horizon finding, and GW extraction. Please note that some of the thorns used in these examples are not (yet) officially part of the Einstein Toolkit and can be found in the [SpacetimeX](https://github.com/EinsteinToolkit/SpacetimeX) repository.

Both CottonmouthBSSNOK and CottonmouthZ4c have a relatively small number of tunable parameters whose default values should work well in many cases. The most important of these are:

1. `CottonmouthBSSNOK::eta_B / CottonmouthZ4c::eta_beta`: This parameter controls the $\Gamma$-driver shift damping coefficient $\eta$. Its typically chosen to be of order $1/M$ where $M$ is the total ADM mass of the system.
2. `Cottonmouth(BSSNOK/Z4c)::dissipation_epsilon`: This parameter controls the strength of the Kreiss-Oliger dissipation filters applied during the evolution.
3. `Cottonmouth(BSSNOK/Z4c)::apply_NewRadX`: If `true`, applies radiating outer boundary conditions using `NewRadX`.

Once you pick some initial data and set the above quantities with reasonable values for your problem, you should be good to go!

## Licensing and Attribution
Cottonmouth was authored by Lucas Timotheo Sanches, Max Morris, and Steven R. Brandt.
It is licensed under version 3 of the Affero General Public License (AGPLv3).

If you use Cottonmouth in your research, we ask that you cite this repository. When our paper is published, we will update this section accordingly.
