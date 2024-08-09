# ROCprofiler-SDK:  Application Profiling, Tracing, and Performance Analysis

> [!NOTE]
Note: rocprofiler-sdk is currently considered a `beta` version and is subject to change in future releases

## Overview

ROCProfiler-SDK is AMD’s new and improved tooling infrastructure, providing a hardware-specific low-level performance analysis interface for profiling and tracing GPU compute applications. To see what's changed [Click Here](source/docs/what-is-rocprof-sdk.rst)

## GPU Metrics

- GPU hardware counters
- HIP API tracing
- HIP kernel tracing
- HSA API tracing
- HSA operation tracing
- Marker(ROCtx) tracing
- PC Sampling (Beta)

## Tool Support

rocprofv3 is the command line tool built using the rocprofiler-sdk library and shipped with the ROCm stack. To see details on
the command line options of rocprofv3, please see rocprofv3 user guide
[Click Here](source/docs/how-to/using-rocprofv3.rst)

## Documentation

We make use of doxygen to generate API documentation automatically. The generated document can be found in the following path:

``` bash
<ROCM_PATH>/share/html/rocprofiler-sdk
```

ROCM_PATH by default is /opt/rocm
It can be set by the user in different locations if needed.

## Build and Installation

```bash
git clone https://git@github.com:ROCm/rocprofiler-sdk.git rocprofiler-sdk-source
cmake                                         \
      -B rocprofiler-sdk-build                \
      -D ROCPROFILER_BUILD_TESTS=ON           \
      -D ROCPROFILER_BUILD_SAMPLES=ON         \
      -D CMAKE_INSTALL_PREFIX=/opt/rocm       \
       rocprofiler-sdk-source

cmake --build rocprofiler-sdk-build --target all --parallel 8
```

To install ROCprofiler, run:

```bash
cmake --build rocprofiler-sdk-build --target install
```

Please see the detailed section on build and installation here: [Click Here](source/docs/install/installation.md)

## Support

Please report in the Github Issues.

## Limitations

- Individual XCC mode is not supported.

- By default, PC sampling API is disabled. To use PC sampling. Setting the `ROCPROFILER_PC_SAMPLING_BETA_ENABLED` environment variable grants access to the PC Sampling experimental beta feature. This feature is still under development and may not be completely stable.
    - **Risk Acknowledgment**: By activating this environment variable, you acknowledge and accept the following potential risks:
       - **Hardware Freeze**: This beta feature could cause your hardware to freeze unexpectedly.
       - **Need for Cold Restart**: In the event of a hardware freeze, you may need to perform a cold restart (turning the hardware off and on) to restore normal operations.
    Please use this beta feature cautiously. It may affect your system's stability and performance. Proceed at your own risk.

- At this point, We do not recommend stress-testing the beta implementation.

- Correlation IDs provided by the PC sampling service are verified only for HIP API calls.

- Timestamps in PC sampling records might not be 100% accurate.

- Using PC sampling on multi-threaded applications might fail with `HSA_STATUS_ERROR_EXCEPTION`.Furthermore, if three or more threads launch operations to the same agent, and if PC sampling is enabled, the `HSA_STATUS_ERROR_EXCEPTION` might appear.

> [!WARNING]
> The latest mainline version of AQLprofile can be found at [https://repo.radeon.com/rocm/misc/aqlprofile/](https://repo.radeon.com/rocm/misc/aqlprofile/). However, it's important to note that updates to the public AQLProfile may not occur as frequently as updates to the rocprofiler-sdk. This discrepancy could lead to a potential mismatch between the AQLprofile binary and the rocprofiler-sdk source.
