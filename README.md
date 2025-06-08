# Cooldata

This Python dataset library is designed to streamline the end-to-end model training process, enabling efficient loading, visualization, and preparation of 3D data for machine learning applications. It supports advanced techniques, including graph neural networks and voxelized methods, with seamless integration into PyTorch workflows.

## Features

- **Data Storage:** Organized in folders containing `.cgns` files for compatibility with computational fluid dynamics tools.
- **PyVista Integration:** Access to dataset samples as PyVista objects for easy 3D visualization and manipulation.
- **Graph Neural Network Support:**
  - **DGL Support:**
    - Surface and volume data in mesh format.
    - 3D visualization of samples and predictions.
    - L2 loss computation and aggregate force evaluation for model training.
  - **Planned PyG Support:** Implementing functionalities similar to DGL.
- **Hugging Face Integration:** Direct dataset loading from [Hugging Face](https://huggingface.co/).
- **Voxelized Flow Field Support:** Facilitates image processing-based ML approaches (Planned).
- **Comprehensive Metadata Accessibility:** For advanced model comparison and evaluation (Planned).

## Installation

Run

```bash
pip install cooldata
```

If you want to use the DGL support, you also need to install the [DGL](https://www.dgl.ai/) library, as documented [here](https://www.dgl.ai/pages/start.html).

## Example Usage

See the `examples` folder for a detailed example of how to use the library.

## Roadmap

- PyG Support
- Re-meshing with Random Point Sampling
- Voxelized Flow Field Support
- Inference of Surface Quantities from Volumetric Predictions
- Enhanced Metadata Accessibility

## Development

This package uses [uv](https://docs.astral.sh/uv/) for package management. To get started, first install uv. Then run

```bash
uv venv
uv sync
```

to create a virtualenv and install the required dependencies in it. For dgl, run the [install script](examples/meshgraphnet/install.sh).
