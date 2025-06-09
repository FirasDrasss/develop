# Cooldata - A Large-Scale Electronics Cooling 3D Flow Field Dataset

Cooldata is a large-scale electronics cooling dataset, containing over 60k stationary 3D flow fields for a diverse set of geometries, simulated with the commercial solver Simcenter STAR-CCM+. This library can be used to acccess the dataset and streamline its application in machine learning tasks.

![example case](docs/_static/case.png)

Find the documentation at [cooldata.readthedocs.io](https://cooldata.readthedocs.io/).

## Features

- **Data Storage:** Organized in folders containing `.cgns` files for compatibility with computational fluid dynamics tools.
- **PyVista Integration:** Access to dataset samples as PyVista objects for easy 3D visualization and manipulation.
- **Graph Neural Network Support:**
  - **DGL Support:**
    - Surface and volume data in mesh format.
    - 3D visualization of samples and predictions.
    - L2 loss computation and aggregate force evaluation for model training.
  - **PyG Support:** Implementing functionalities similar to DGL.
- **Hugging Face Integration:** Direct dataset loading from [Hugging Face](https://huggingface.co/).
- **Voxelized Flow Field Support:** Facilitates image processing-based ML approaches.
- **Comprehensive Metadata Accessibility:** All metadata is accessible through the library.

## Installation

Run

```bash
pip install cooldata
```

If you want to use the DGL support, you also need to install the [DGL](https://www.dgl.ai/) library, as documented [here](https://www.dgl.ai/pages/start.html).

## Example Usage

See the `examples` folder for a detailed example of how to use the library.

## Roadmap

- Re-meshing with Random Point Sampling
- Inference of Surface Quantities from Volumetric Predictions
