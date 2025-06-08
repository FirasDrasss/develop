.. Dataset Library for 3D Machine Learning documentation master file, created by
   sphinx-quickstart on Fri May 23 16:36:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CoolData: Dataset Library for 3D Machine Learning
=====================================================


This Python dataset library is designed to **streamline the end-to-end model training process**, enabling efficient 
- loading,
- visualization, 
- preparation of 3D data for machine learning applications. 
It supports advanced techniques, including graph neural networks and voxelized methods, with seamless integration into PyTorch workflows.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/example_metadata
   usage/autodoc

Features
---------
- **Data Storage:** Organized in folders containing .cgns files for compatibility with computational fluid dynamics tools.
- **PyVista Integration:** Access to dataset samples as PyVista objects for easy 3D visualization and manipulation.
- **Graph Neural Network Support:**

   - DGL Support:

      - Surface and volume data in mesh format.
      - 3D visualization of samples and predictions.
      - L2 loss computation and aggregate force evaluation for model training.

   - PyG Support: Implementing functionalities similar to DGL *(Planned)*.

- **Hugging Face Integration:** Direct dataset loading from `Hugging Face <https://huggingface.co/>`_.
- Voxelized Flow Field Support: Facilitates image processing-based ML approaches *(Planned)*.
- Comprehensive Metadata Accessibility: For advanced model comparison and evaluation *(Planned)*.


Installation
--------------
``pip install cooldata``

If you want to use the DGL support, you also need to install the `DGL library <https://www.dgl.ai/>`_, as documented `here <https://www.dgl.ai/pages/start.html>`_.
