.. meta::
  :description: Documentation of the installation, configuration, use of the ROCprofiler SDK, and rocprofv3 command-line tool 
  :keywords: ROCprofiler-SDK tool, ROCprofiler-SDK library, rocprofv3, ROCm, API, reference

.. _index:

******************************************
ROCprofiler-SDK documentation
******************************************

ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software.
It supports application tracing to provide a big picture of the GPU application execution and kernel profiling to provide low-level hardware details from the performance counters.
The ROCprofiler-SDK library provides runtime-independent APIs for tracing runtime calls and asynchronous activities such as GPU kernel dispatches and memory moves. The tracing includes callback APIs for runtime API tracing and activity APIs for asynchronous activity records logging. 

In summary, ROCprofiler-SDK combines `ROCProfiler <https://rocm.docs.amd.com/projects/rocprofiler/en/latest/index.html>`_ and `ROCTracer <https://rocm.docs.amd.com/projects/roctracer/en/latest/index.html>`_.
You can utilize the ROCprofiler-SDK to develop a tool for profiling and tracing HIP applications on ROCm software.

The code is open and hosted at `<https://github.com/ROCm/rocprofiler-sdk>`_.

.. note::
  ROCprofiler-SDK is in beta and subject to change in future releases.

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Installation <install/installation>`

  .. grid-item-card:: How to

    * :ref:`using-rocprofv3`
    * :doc:`Samples <how-to/samples>`
    
  .. grid-item-card:: API reference

    * :doc:`Buffered services <api-reference/buffered_services>`
    * :doc:`Callback services <api-reference/callback_services>`
    * :doc:`Counter collection services <api-reference/counter_collection_services>`
    * :doc:`Intercept table <api-reference/intercept_table>`
    * :doc:`PC sampling <api-reference/pc_sampling>`
    * :doc:`Tool library <api-reference/tool_library>`
    * :doc:`API library <_doxygen/html/index>`

  .. grid-item-card:: Conceptual

    * :ref:`comparing-with-legacy-tools`
    
To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
