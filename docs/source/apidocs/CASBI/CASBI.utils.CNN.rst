:py:mod:`CASBI.utils.CNN`
=========================

.. py:module:: CASBI.utils.CNN

.. autodoc2-docstring:: CASBI.utils.CNN
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ConvNet_halo <CASBI.utils.CNN.ConvNet_halo>`
     - .. autodoc2-docstring:: CASBI.utils.CNN.ConvNet_halo
          :summary:
   * - :py:obj:`ConvNet_subhalo <CASBI.utils.CNN.ConvNet_subhalo>`
     - .. autodoc2-docstring:: CASBI.utils.CNN.ConvNet_subhalo
          :summary:

API
~~~

.. py:class:: ConvNet_halo(output_dim)
   :canonical: CASBI.utils.CNN.ConvNet_halo

   Bases: :py:obj:`torch.nn.Module`

   .. autodoc2-docstring:: CASBI.utils.CNN.ConvNet_halo

   .. rubric:: Initialization

   .. autodoc2-docstring:: CASBI.utils.CNN.ConvNet_halo.__init__

   .. py:method:: set_device(device)
      :canonical: CASBI.utils.CNN.ConvNet_halo.set_device

      .. autodoc2-docstring:: CASBI.utils.CNN.ConvNet_halo.set_device

   .. py:method:: forward(x)
      :canonical: CASBI.utils.CNN.ConvNet_halo.forward

      .. autodoc2-docstring:: CASBI.utils.CNN.ConvNet_halo.forward

.. py:class:: ConvNet_subhalo(output_dim)
   :canonical: CASBI.utils.CNN.ConvNet_subhalo

   Bases: :py:obj:`torch.nn.Module`

   .. autodoc2-docstring:: CASBI.utils.CNN.ConvNet_subhalo

   .. rubric:: Initialization

   .. autodoc2-docstring:: CASBI.utils.CNN.ConvNet_subhalo.__init__

   .. py:method:: set_device(device)
      :canonical: CASBI.utils.CNN.ConvNet_subhalo.set_device

      .. autodoc2-docstring:: CASBI.utils.CNN.ConvNet_subhalo.set_device

   .. py:method:: forward(x)
      :canonical: CASBI.utils.CNN.ConvNet_subhalo.forward

      .. autodoc2-docstring:: CASBI.utils.CNN.ConvNet_subhalo.forward
