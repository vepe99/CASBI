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

   * - :py:obj:`CustomDataset_halo <CASBI.inference.CustomDataset_halo>`
     - .. autodoc2-docstring:: CASBI.inference.CustomDataset_halo
          :summary:
   * - :py:obj:`CustomDataset_subhalo <CASBI.inference.CustomDataset_subhalo>`
     - .. autodoc2-docstring:: CASBI.inference.CustomDataset_subhalo
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

.. py:class:: CustomDataset_halo(observation, parameters, device)
   :canonical: CASBI.inference.CustomDataset_halo

   Bases: :py:obj:`torch.utils.data.Dataset`

   .. autodoc2-docstring:: CASBI.inference.CustomDataset_halo

   .. rubric:: Initialization

   .. autodoc2-docstring:: CASBI.inference.CustomDataset_halo.__init__

   .. py:method:: __len__()
      :canonical: CASBI.inference.CustomDataset_halo.__len__

      .. autodoc2-docstring:: CASBI.inference.CustomDataset_halo.__len__

   .. py:method:: __getitem__(idx)
      :canonical: CASBI.inference.CustomDataset_halo.__getitem__

      .. autodoc2-docstring:: CASBI.inference.CustomDataset_halo.__getitem__

.. py:class:: CustomDataset_subhalo(observation, parameters, device)
   :canonical: CASBI.inference.CustomDataset_subhalo

   Bases: :py:obj:`torch.utils.data.Dataset`

   .. autodoc2-docstring:: CASBI.inference.CustomDataset_subhalo

   .. rubric:: Initialization

   .. autodoc2-docstring:: CASBI.inference.CustomDataset_subhalo.__init__

   .. py:method:: __len__()
      :canonical: CASBI.inference.CustomDataset_subhalo.__len__

      .. autodoc2-docstring:: CASBI.inference.CustomDataset_subhalo.__len__

   .. py:method:: __getitem__(idx)
      :canonical: CASBI.inference.CustomDataset_subhalo.__getitem__

      .. autodoc2-docstring:: CASBI.inference.CustomDataset_subhalo.__getitem__

.. py:function:: train_inference(x: torch.Tensor, theta: torch.Tensor, validation_fraction: float = 0.2, output_dir: str = './', device: str = 'cuda', N_nets=4, hidden_feature: int = 100, num_transforms: int = 20, model: str = 'nsf', embedding_net: str = ConvNet_halo(output_dim=32), custom_dataset: torch.utils.data.Dataset = CustomDataset_halo, custom_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader, minimum_theta: list = [3.5, -2.0], maximum_theta: list = [10, 1.15], batch_size: int = 2048, learning_rate: float = 1e-05, stop_after_epochs: int = 20, norm_x=False, norm_theta=True)
   :canonical: CASBI.inference.train_inference

   .. autodoc2-docstring:: CASBI.inference.train_inference
