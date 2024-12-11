:py:mod:`CASBI.inference`
=========================

.. py:module:: CASBI.inference

.. autodoc2-docstring:: CASBI.inference
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`CustomDataset <CASBI.inference.CustomDataset>`
     - .. autodoc2-docstring:: CASBI.inference.CustomDataset
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`train_inference <CASBI.inference.train_inference>`
     - .. autodoc2-docstring:: CASBI.inference.train_inference
          :summary:

API
~~~

.. py:class:: CustomDataset(observation, parameters)
   :canonical: CASBI.inference.CustomDataset

   Bases: :py:obj:`torch.utils.data.Dataset`

   .. autodoc2-docstring:: CASBI.inference.CustomDataset

   .. rubric:: Initialization

   .. autodoc2-docstring:: CASBI.inference.CustomDataset.__init__

   .. py:method:: __len__()
      :canonical: CASBI.inference.CustomDataset.__len__

      .. autodoc2-docstring:: CASBI.inference.CustomDataset.__len__

   .. py:method:: __getitem__(idx)
      :canonical: CASBI.inference.CustomDataset.__getitem__

      .. autodoc2-docstring:: CASBI.inference.CustomDataset.__getitem__

.. py:function:: train_inference(x: torch.Tensor, theta: torch.Tensor, validation_fraction: float = 0.2, output_dir: str = './', device: str = 'cuda', N_nets=4, hidden_feature: int = 100, num_transforms: int = 20, model: str = 'nsf', embedding_net: str = ConvNet(output_dim=32), minimum_theta: list = [3.5, -2.0], maximum_theta: list = [10, 1.15], batch_size: int = 1024, learning_rate: float = 1e-05, stop_after_epochs: int = 20)
   :canonical: CASBI.inference.train_inference

   .. autodoc2-docstring:: CASBI.inference.train_inference
