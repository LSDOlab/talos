# TALOS: A toolbox for spacecraft conceptual design

## Abstract

We present the Toolbox for Analysis and Large-scale Optimization of
Spacecraft (TALOS), a framework designed for applying large-scale
multidisciplinary design optimization (MDO) to spacecraft design
problems. The framework is built using the Computational System Design
Language (CSDL), with abstractions for users to describe systems at a
high level. CSDL is a compiled, embedded domain-specific language that
fully automates derivative computation using the adjoint method. CSDL
provides a unified interface for defining MDO problems, separating model
definition from low-level program implementation details. TALOS provides
discipline models for spacecraft mission designers to perform analyses,
optimizations, and trade studies early in the design process. TALOS also
provides interfaces for users to provide high-level system descriptions
without the need to use CSDL directly, which simplifies the exploration
of different spacecraft configurations. We describe the interfaces in
TALOS available to users and run analyses on selected spacecraft
subsystem disciplines to demonstrate the current capabilities of TALOS. 

## Paper

[arXiv](https://arxiv.org/abs/2303.14936)

## Citing TALOS

```
@misc{gandarillas2023talos,
      title={TALOS: A toolbox for spacecraft conceptual design}, 
      author={Victor Gandarillas and John T. Hwang},
      year={2023},
      eprint={2303.14936},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```