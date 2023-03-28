## csdl_future

This directory contains classes that should be added to CSDL, plus
custom operations that should be implemented in CSDL compiler back ends.

Each class that should be added to CSDL enables users to create
surrogate models by importing training data and setting parameters
specific to the type of surrogate model in use without the need to
implement the surrogate model or call any helper functions to train the
model.

TODO: how to use surrogate models for validation at compile time?

TODO: make `smt` a dependency of CSDL compiler back ends instead of
implicitly requiring users to use `smt` within `CustomOperation` classes

TODO: add docs for CSDL surrogate modeling