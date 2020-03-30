.. StarDistPyTorch documentation master file, created by
   sphinx-quickstart on Mon Mar 30 02:20:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to StarDistPyTorch's documentation!
===========================================
**StarDistPyTorch** is an attempt to reimplement **StarDist** in **PyTorch**. Notable differences in this implementation include:

* New function to patch images
* Slightly different loss function to incorporate class imbalance (In *StarDistPyTorch*, background pixels are weighted 10 times less than foreground pixels during the evaluation of the *Binary Cross Entropy Loss*)
* Grid is *not* downscaled (*StarDist* predicted on a downscaled-image to save memory)
* `self.train_foreground_only` parameter, which is used for effective sampling of patches during training in *StarDist*, is not implemented in *StarDistPyTorch*.
* Shape completion is not implemented in *StarDistPyTorch*

Potential TODOs on the horizon include:

.. todo::
   
   | Adding functionality for *shape completion*
   | Adding functionality for evaluating *field of view*  

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Installation
==================
* Open a new terminal and type in the following commands inside a preferred directory location

.. code-block:: bash

   git clone https://github.com/MLbyML/StarDistPyTorch.git
   git checkout BaseImplementation
   conda env create -f environment.yml
   conda activate stardistPytorchEnv
   python3 -m ipykernel install --user --name stardistPytorchEnv --display-name "stardistPytorchEnv"

   
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
