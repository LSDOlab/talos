# Preparing a CubeSat CAD Model to Generate a Solar Power Model for OpenMDAO

We use raytracing to compute the amount of solar exposure of a CubeSat's
solar panels as a function of azimuth and elevation.
Raytracing allows us to take into account shadows cast from one part of
the CubeSat onto a solar panel.
The result of the raytracing algorithm is a table mapping values of
azimuth and elevation to exposure data.
We use a BSpline interpolant to construct a continuous and
differentiable model suitable for use with gradient-based approaches to
optimization.
The BSpline model of the exposure as a continuous and differentiable
function of azimuth and elevation is then inorporated into our OpenMDAO
model of each CubeSat.

## Generating Data for SMT

### Assumptions in the Guide

This guide assumes that you have an STL file of a CubeSat.
You should be able to open your CubeSat assembly in a CAD program and
export an STL part file.

### STL Conversion to PLY

We use face colors to designate faces that correspond to solar panels.
STL files do not encode colors for faces or vertices, so we will use PLY
files.
The script we use to compute solar exposure only works with PLY files.
PLY files are also encoded in ASCII format, making their changes easy to
track in a VCS/SCM.

### Face Coloring

[Add colors to face](https://blender.stackexchange.com/questions/516/add-different-materials-to-different-parts-of-a-mesh)

> Note: You will need to click, "Use Nodes"

Export to OBJ file.
Make sure you are only exporting the selected object in your Blender
scene.

Open the OBJ file in MeshLab and export to PLY.
Make sure all the checkboxes are checked before exporting, especially
colors.

Then run the script to generate the table.

## Building a Model using SMT

Inputs are azimuth and elevation.
Outputs are exposure.
We will use azimuth, elevation, and exposure to generate a model with
SMT.
