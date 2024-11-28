:py:mod:`CASBI.create_template_library`
=======================================

.. py:module:: CASBI.create_template_library

.. autodoc2-docstring:: CASBI.create_template_library
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`TemplateLibrary <CASBI.create_template_library.TemplateLibrary>`
     - .. autodoc2-docstring:: CASBI.create_template_library.TemplateLibrary
          :summary:

API
~~~

.. py:class:: TemplateLibrary(galaxy_file_path: str, dataframe_path: str, preprocessing_path: str, M_tot: float = 1410000000.0, alpha=1.25, d: float = 0.1, bins: int = 64, sigma: float = 0.0, perc_feh: float = 0.1, perc_ofe: float = 0.1, galaxy_names_to_remove: list = ['g6.31e09.01024', 'g6.31e09.00832', 'g6.31e09.00704', 'g6.31e09.00768', 'g6.31e09.00960', 'g6.31e09.00896'])
   :canonical: CASBI.create_template_library.TemplateLibrary

   .. autodoc2-docstring:: CASBI.create_template_library.TemplateLibrary

   .. rubric:: Initialization

   .. autodoc2-docstring:: CASBI.create_template_library.TemplateLibrary.__init__

   .. py:method:: pdf(m, m_max, m_min, alpha)
      :canonical: CASBI.create_template_library.TemplateLibrary.pdf

      .. autodoc2-docstring:: CASBI.create_template_library.TemplateLibrary.pdf

   .. py:method:: cdf(m, m_max, m_min, alpha)
      :canonical: CASBI.create_template_library.TemplateLibrary.cdf

      .. autodoc2-docstring:: CASBI.create_template_library.TemplateLibrary.cdf

   .. py:method:: inverse_cdf(y, m_max, m_min, alpha)
      :canonical: CASBI.create_template_library.TemplateLibrary.inverse_cdf

      .. autodoc2-docstring:: CASBI.create_template_library.TemplateLibrary.inverse_cdf

   .. py:method:: gen_subhalo_sample(samples, masses, times, nbrs)
      :canonical: CASBI.create_template_library.TemplateLibrary.gen_subhalo_sample

      .. autodoc2-docstring:: CASBI.create_template_library.TemplateLibrary.gen_subhalo_sample

   .. py:method:: gen_halo(j, galaxies_test=None)
      :canonical: CASBI.create_template_library.TemplateLibrary.gen_halo

      .. autodoc2-docstring:: CASBI.create_template_library.TemplateLibrary.gen_halo

   .. py:method:: gen_libary(N_test, N_train)
      :canonical: CASBI.create_template_library.TemplateLibrary.gen_libary

      .. autodoc2-docstring:: CASBI.create_template_library.TemplateLibrary.gen_libary

   .. py:method:: get_inference_input()
      :canonical: CASBI.create_template_library.TemplateLibrary.get_inference_input

      .. autodoc2-docstring:: CASBI.create_template_library.TemplateLibrary.get_inference_input
