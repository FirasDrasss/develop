.. Dataset Library for 3D Machine Learning documentation master file, created by
   sphinx-quickstart on Fri May 23 16:36:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cooldata - A Large-Scale Electronics Cooling 3D Flow Field Dataset
==================================================================


Cooldata is a large-scale electronics cooling dataset, containing over 60k stationary 3D flow fields for a diverse set of geometries, simulated with the commercial solver Simcenter STAR-CCM+. This library can be used to acccess the dataset and streamline its application in machine learning tasks.

.. image:: _static/case.png
   :target: _static/case.png
   :alt: A sample case from the Cooldata dataset

Features
---------

- **Data Storage:** Organized in folders containing `.cgns` files for compatibility with computational fluid dynamics tools.
- **PyVista Integration:** Access to dataset samples as PyVista objects for easy 3D visualization and manipulation.
- **Graph Neural Network Support:**
- **DGL Support:** Surface and volume data in mesh format, 3D visualization of samples and predictions, L2 loss computation and aggregate force evaluation for model training.
- **PyG Support:** Implementing functionalities similar to DGL.
- **Hugging Face Integration:** Direct dataset loading from [Hugging Face](https://huggingface.co/).
- **Voxelized Flow Field Support:** Facilitates image processing-based ML approaches.
- **Comprehensive Metadata Accessibility:** All metadata is accessible through the library.

Installation
----------------
Run

``pip install cooldata``

If you want to use the DGL support, you also need to install the [DGL](https://www.dgl.ai/) library, as documented [here](https://www.dgl.ai/pages/start.html).



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/getting_started
   usage/voxels
   usage/dgl
   usage/pyg
   api-reference/cooldata
