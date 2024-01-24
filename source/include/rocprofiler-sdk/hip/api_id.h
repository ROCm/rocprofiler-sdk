// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

/**
 * @brief ROCProfiler enumeration of HIP API tracing operations
 */
typedef enum  // NOLINT(performance-enum-size)
{
    ROCPROFILER_HIP_API_ID_NONE       = -1,
    ROCPROFILER_HIP_API_ID_hipApiName = 0,
    ROCPROFILER_HIP_API_ID_hipArray3DCreate,
    ROCPROFILER_HIP_API_ID_hipArray3DGetDescriptor,
    ROCPROFILER_HIP_API_ID_hipArrayCreate,
    ROCPROFILER_HIP_API_ID_hipArrayDestroy,
    ROCPROFILER_HIP_API_ID_hipArrayGetDescriptor,
    ROCPROFILER_HIP_API_ID_hipArrayGetInfo,
    ROCPROFILER_HIP_API_ID_hipBindTexture,                  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipBindTexture2D,                // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipBindTextureToArray,           // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipBindTextureToMipmappedArray,  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipChooseDevice,
    ROCPROFILER_HIP_API_ID_hipChooseDeviceR0000,
    ROCPROFILER_HIP_API_ID_hipConfigureCall,
    ROCPROFILER_HIP_API_ID_hipCreateSurfaceObject,
    ROCPROFILER_HIP_API_ID_hipCreateTextureObject,  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipCtxCreate,
    ROCPROFILER_HIP_API_ID_hipCtxDestroy,
    ROCPROFILER_HIP_API_ID_hipCtxDisablePeerAccess,
    ROCPROFILER_HIP_API_ID_hipCtxEnablePeerAccess,
    ROCPROFILER_HIP_API_ID_hipCtxGetApiVersion,
    ROCPROFILER_HIP_API_ID_hipCtxGetCacheConfig,
    ROCPROFILER_HIP_API_ID_hipCtxGetCurrent,
    ROCPROFILER_HIP_API_ID_hipCtxGetDevice,
    ROCPROFILER_HIP_API_ID_hipCtxGetFlags,
    ROCPROFILER_HIP_API_ID_hipCtxGetSharedMemConfig,
    ROCPROFILER_HIP_API_ID_hipCtxPopCurrent,
    ROCPROFILER_HIP_API_ID_hipCtxPushCurrent,
    ROCPROFILER_HIP_API_ID_hipCtxSetCacheConfig,
    ROCPROFILER_HIP_API_ID_hipCtxSetCurrent,
    ROCPROFILER_HIP_API_ID_hipCtxSetSharedMemConfig,
    ROCPROFILER_HIP_API_ID_hipCtxSynchronize,
    ROCPROFILER_HIP_API_ID_hipDestroyExternalMemory,
    ROCPROFILER_HIP_API_ID_hipDestroyExternalSemaphore,
    ROCPROFILER_HIP_API_ID_hipDestroySurfaceObject,
    ROCPROFILER_HIP_API_ID_hipDestroyTextureObject,  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipDeviceCanAccessPeer,
    ROCPROFILER_HIP_API_ID_hipDeviceComputeCapability,
    ROCPROFILER_HIP_API_ID_hipDeviceDisablePeerAccess,
    ROCPROFILER_HIP_API_ID_hipDeviceEnablePeerAccess,
    ROCPROFILER_HIP_API_ID_hipDeviceGet,
    ROCPROFILER_HIP_API_ID_hipDeviceGetAttribute,
    ROCPROFILER_HIP_API_ID_hipDeviceGetByPCIBusId,
    ROCPROFILER_HIP_API_ID_hipDeviceGetCacheConfig,
    ROCPROFILER_HIP_API_ID_hipDeviceGetDefaultMemPool,
    ROCPROFILER_HIP_API_ID_hipDeviceGetGraphMemAttribute,
    ROCPROFILER_HIP_API_ID_hipDeviceGetLimit,
    ROCPROFILER_HIP_API_ID_hipDeviceGetMemPool,
    ROCPROFILER_HIP_API_ID_hipDeviceGetName,
    ROCPROFILER_HIP_API_ID_hipDeviceGetP2PAttribute,
    ROCPROFILER_HIP_API_ID_hipDeviceGetPCIBusId,
    ROCPROFILER_HIP_API_ID_hipDeviceGetSharedMemConfig,
    ROCPROFILER_HIP_API_ID_hipDeviceGetStreamPriorityRange,
    ROCPROFILER_HIP_API_ID_hipDeviceGetUuid,
    ROCPROFILER_HIP_API_ID_hipDeviceGraphMemTrim,
    ROCPROFILER_HIP_API_ID_hipDevicePrimaryCtxGetState,
    ROCPROFILER_HIP_API_ID_hipDevicePrimaryCtxRelease,
    ROCPROFILER_HIP_API_ID_hipDevicePrimaryCtxReset,
    ROCPROFILER_HIP_API_ID_hipDevicePrimaryCtxRetain,
    ROCPROFILER_HIP_API_ID_hipDevicePrimaryCtxSetFlags,
    ROCPROFILER_HIP_API_ID_hipDeviceReset,
    ROCPROFILER_HIP_API_ID_hipDeviceSetCacheConfig,
    ROCPROFILER_HIP_API_ID_hipDeviceSetGraphMemAttribute,
    ROCPROFILER_HIP_API_ID_hipDeviceSetLimit,
    ROCPROFILER_HIP_API_ID_hipDeviceSetMemPool,
    ROCPROFILER_HIP_API_ID_hipDeviceSetSharedMemConfig,
    ROCPROFILER_HIP_API_ID_hipDeviceSynchronize,
    ROCPROFILER_HIP_API_ID_hipDeviceTotalMem,
    ROCPROFILER_HIP_API_ID_hipDriverGetVersion,
    ROCPROFILER_HIP_API_ID_hipDrvGetErrorName,
    ROCPROFILER_HIP_API_ID_hipDrvGetErrorString,
    ROCPROFILER_HIP_API_ID_hipDrvGraphAddMemcpyNode,
    ROCPROFILER_HIP_API_ID_hipDrvMemcpy2DUnaligned,
    ROCPROFILER_HIP_API_ID_hipDrvMemcpy3D,
    ROCPROFILER_HIP_API_ID_hipDrvMemcpy3DAsync,
    ROCPROFILER_HIP_API_ID_hipDrvPointerGetAttributes,
    ROCPROFILER_HIP_API_ID_hipEventCreate,
    ROCPROFILER_HIP_API_ID_hipEventCreateWithFlags,
    ROCPROFILER_HIP_API_ID_hipEventDestroy,
    ROCPROFILER_HIP_API_ID_hipEventElapsedTime,
    ROCPROFILER_HIP_API_ID_hipEventQuery,
    ROCPROFILER_HIP_API_ID_hipEventRecord,
    ROCPROFILER_HIP_API_ID_hipEventSynchronize,
    ROCPROFILER_HIP_API_ID_hipExtGetLinkTypeAndHopCount,
    ROCPROFILER_HIP_API_ID_hipExtLaunchKernel,
    ROCPROFILER_HIP_API_ID_hipExtLaunchMultiKernelMultiDevice,
    ROCPROFILER_HIP_API_ID_hipExtMallocWithFlags,
    ROCPROFILER_HIP_API_ID_hipExtStreamCreateWithCUMask,
    ROCPROFILER_HIP_API_ID_hipExtStreamGetCUMask,
    ROCPROFILER_HIP_API_ID_hipExternalMemoryGetMappedBuffer,
    ROCPROFILER_HIP_API_ID_hipFree,
    ROCPROFILER_HIP_API_ID_hipFreeArray,
    ROCPROFILER_HIP_API_ID_hipFreeAsync,
    ROCPROFILER_HIP_API_ID_hipFreeHost,
    ROCPROFILER_HIP_API_ID_hipFreeMipmappedArray,
    ROCPROFILER_HIP_API_ID_hipFuncGetAttribute,
    ROCPROFILER_HIP_API_ID_hipFuncGetAttributes,
    ROCPROFILER_HIP_API_ID_hipFuncSetAttribute,
    ROCPROFILER_HIP_API_ID_hipFuncSetCacheConfig,
    ROCPROFILER_HIP_API_ID_hipFuncSetSharedMemConfig,
    ROCPROFILER_HIP_API_ID_hipGLGetDevices,
    ROCPROFILER_HIP_API_ID_hipGetChannelDesc,
    ROCPROFILER_HIP_API_ID_hipGetDevice,
    ROCPROFILER_HIP_API_ID_hipGetDeviceCount,
    ROCPROFILER_HIP_API_ID_hipGetDeviceFlags,
    ROCPROFILER_HIP_API_ID_hipGetDevicePropertiesR0600,
    ROCPROFILER_HIP_API_ID_hipGetDevicePropertiesR0000,
    ROCPROFILER_HIP_API_ID_hipGetErrorName,
    ROCPROFILER_HIP_API_ID_hipGetErrorString,
    ROCPROFILER_HIP_API_ID_hipGetLastError,
    ROCPROFILER_HIP_API_ID_hipGetMipmappedArrayLevel,
    ROCPROFILER_HIP_API_ID_hipGetSymbolAddress,
    ROCPROFILER_HIP_API_ID_hipGetSymbolSize,
    ROCPROFILER_HIP_API_ID_hipGetTextureAlignmentOffset,         // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipGetTextureObjectResourceDesc,      // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipGetTextureObjectResourceViewDesc,  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipGetTextureObjectTextureDesc,       // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipGetTextureReference,               // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipGraphAddChildGraphNode,
    ROCPROFILER_HIP_API_ID_hipGraphAddDependencies,
    ROCPROFILER_HIP_API_ID_hipGraphAddEmptyNode,
    ROCPROFILER_HIP_API_ID_hipGraphAddEventRecordNode,
    ROCPROFILER_HIP_API_ID_hipGraphAddEventWaitNode,
    ROCPROFILER_HIP_API_ID_hipGraphAddHostNode,
    ROCPROFILER_HIP_API_ID_hipGraphAddKernelNode,
    ROCPROFILER_HIP_API_ID_hipGraphAddMemAllocNode,
    ROCPROFILER_HIP_API_ID_hipGraphAddMemFreeNode,
    ROCPROFILER_HIP_API_ID_hipGraphAddMemcpyNode,
    ROCPROFILER_HIP_API_ID_hipGraphAddMemcpyNode1D,
    ROCPROFILER_HIP_API_ID_hipGraphAddMemcpyNodeFromSymbol,
    ROCPROFILER_HIP_API_ID_hipGraphAddMemcpyNodeToSymbol,
    ROCPROFILER_HIP_API_ID_hipGraphAddMemsetNode,
    ROCPROFILER_HIP_API_ID_hipGraphChildGraphNodeGetGraph,
    ROCPROFILER_HIP_API_ID_hipGraphClone,
    ROCPROFILER_HIP_API_ID_hipGraphCreate,
    ROCPROFILER_HIP_API_ID_hipGraphDebugDotPrint,
    ROCPROFILER_HIP_API_ID_hipGraphDestroy,
    ROCPROFILER_HIP_API_ID_hipGraphDestroyNode,
    ROCPROFILER_HIP_API_ID_hipGraphEventRecordNodeGetEvent,
    ROCPROFILER_HIP_API_ID_hipGraphEventRecordNodeSetEvent,
    ROCPROFILER_HIP_API_ID_hipGraphEventWaitNodeGetEvent,
    ROCPROFILER_HIP_API_ID_hipGraphEventWaitNodeSetEvent,
    ROCPROFILER_HIP_API_ID_hipGraphExecChildGraphNodeSetParams,
    ROCPROFILER_HIP_API_ID_hipGraphExecDestroy,
    ROCPROFILER_HIP_API_ID_hipGraphExecEventRecordNodeSetEvent,
    ROCPROFILER_HIP_API_ID_hipGraphExecEventWaitNodeSetEvent,
    ROCPROFILER_HIP_API_ID_hipGraphExecHostNodeSetParams,
    ROCPROFILER_HIP_API_ID_hipGraphExecKernelNodeSetParams,
    ROCPROFILER_HIP_API_ID_hipGraphExecMemcpyNodeSetParams,
    ROCPROFILER_HIP_API_ID_hipGraphExecMemcpyNodeSetParams1D,
    ROCPROFILER_HIP_API_ID_hipGraphExecMemcpyNodeSetParamsFromSymbol,
    ROCPROFILER_HIP_API_ID_hipGraphExecMemcpyNodeSetParamsToSymbol,
    ROCPROFILER_HIP_API_ID_hipGraphExecMemsetNodeSetParams,
    ROCPROFILER_HIP_API_ID_hipGraphExecUpdate,
    ROCPROFILER_HIP_API_ID_hipGraphGetEdges,
    ROCPROFILER_HIP_API_ID_hipGraphGetNodes,
    ROCPROFILER_HIP_API_ID_hipGraphGetRootNodes,
    ROCPROFILER_HIP_API_ID_hipGraphHostNodeGetParams,
    ROCPROFILER_HIP_API_ID_hipGraphHostNodeSetParams,
    ROCPROFILER_HIP_API_ID_hipGraphInstantiate,
    ROCPROFILER_HIP_API_ID_hipGraphInstantiateWithFlags,
    ROCPROFILER_HIP_API_ID_hipGraphKernelNodeCopyAttributes,
    ROCPROFILER_HIP_API_ID_hipGraphKernelNodeGetAttribute,
    ROCPROFILER_HIP_API_ID_hipGraphKernelNodeGetParams,
    ROCPROFILER_HIP_API_ID_hipGraphKernelNodeSetAttribute,
    ROCPROFILER_HIP_API_ID_hipGraphKernelNodeSetParams,
    ROCPROFILER_HIP_API_ID_hipGraphLaunch,
    ROCPROFILER_HIP_API_ID_hipGraphMemAllocNodeGetParams,
    ROCPROFILER_HIP_API_ID_hipGraphMemFreeNodeGetParams,
    ROCPROFILER_HIP_API_ID_hipGraphMemcpyNodeGetParams,
    ROCPROFILER_HIP_API_ID_hipGraphMemcpyNodeSetParams,
    ROCPROFILER_HIP_API_ID_hipGraphMemcpyNodeSetParams1D,
    ROCPROFILER_HIP_API_ID_hipGraphMemcpyNodeSetParamsFromSymbol,
    ROCPROFILER_HIP_API_ID_hipGraphMemcpyNodeSetParamsToSymbol,
    ROCPROFILER_HIP_API_ID_hipGraphMemsetNodeGetParams,
    ROCPROFILER_HIP_API_ID_hipGraphMemsetNodeSetParams,
    ROCPROFILER_HIP_API_ID_hipGraphNodeFindInClone,
    ROCPROFILER_HIP_API_ID_hipGraphNodeGetDependencies,
    ROCPROFILER_HIP_API_ID_hipGraphNodeGetDependentNodes,
    ROCPROFILER_HIP_API_ID_hipGraphNodeGetEnabled,
    ROCPROFILER_HIP_API_ID_hipGraphNodeGetType,
    ROCPROFILER_HIP_API_ID_hipGraphNodeSetEnabled,
    ROCPROFILER_HIP_API_ID_hipGraphReleaseUserObject,
    ROCPROFILER_HIP_API_ID_hipGraphRemoveDependencies,
    ROCPROFILER_HIP_API_ID_hipGraphRetainUserObject,
    ROCPROFILER_HIP_API_ID_hipGraphUpload,
    ROCPROFILER_HIP_API_ID_hipGraphicsGLRegisterBuffer,
    ROCPROFILER_HIP_API_ID_hipGraphicsGLRegisterImage,
    ROCPROFILER_HIP_API_ID_hipGraphicsMapResources,
    ROCPROFILER_HIP_API_ID_hipGraphicsResourceGetMappedPointer,
    ROCPROFILER_HIP_API_ID_hipGraphicsSubResourceGetMappedArray,
    ROCPROFILER_HIP_API_ID_hipGraphicsUnmapResources,
    ROCPROFILER_HIP_API_ID_hipGraphicsUnregisterResource,
    ROCPROFILER_HIP_API_ID_hipHostAlloc,
    ROCPROFILER_HIP_API_ID_hipHostFree,
    ROCPROFILER_HIP_API_ID_hipHostGetDevicePointer,
    ROCPROFILER_HIP_API_ID_hipHostGetFlags,
    ROCPROFILER_HIP_API_ID_hipHostMalloc,
    ROCPROFILER_HIP_API_ID_hipHostRegister,
    ROCPROFILER_HIP_API_ID_hipHostUnregister,
    ROCPROFILER_HIP_API_ID_hipImportExternalMemory,
    ROCPROFILER_HIP_API_ID_hipImportExternalSemaphore,
    ROCPROFILER_HIP_API_ID_hipInit,
    ROCPROFILER_HIP_API_ID_hipIpcCloseMemHandle,
    ROCPROFILER_HIP_API_ID_hipIpcGetEventHandle,
    ROCPROFILER_HIP_API_ID_hipIpcGetMemHandle,
    ROCPROFILER_HIP_API_ID_hipIpcOpenEventHandle,
    ROCPROFILER_HIP_API_ID_hipIpcOpenMemHandle,
    ROCPROFILER_HIP_API_ID_hipKernelNameRef,
    ROCPROFILER_HIP_API_ID_hipKernelNameRefByPtr,
    ROCPROFILER_HIP_API_ID_hipLaunchByPtr,
    ROCPROFILER_HIP_API_ID_hipLaunchCooperativeKernel,
    ROCPROFILER_HIP_API_ID_hipLaunchCooperativeKernelMultiDevice,
    ROCPROFILER_HIP_API_ID_hipLaunchHostFunc,
    ROCPROFILER_HIP_API_ID_hipLaunchKernel,
    ROCPROFILER_HIP_API_ID_hipMalloc,
    ROCPROFILER_HIP_API_ID_hipMalloc3D,
    ROCPROFILER_HIP_API_ID_hipMalloc3DArray,
    ROCPROFILER_HIP_API_ID_hipMallocArray,
    ROCPROFILER_HIP_API_ID_hipMallocAsync,
    ROCPROFILER_HIP_API_ID_hipMallocFromPoolAsync,
    ROCPROFILER_HIP_API_ID_hipMallocHost,
    ROCPROFILER_HIP_API_ID_hipMallocManaged,
    ROCPROFILER_HIP_API_ID_hipMallocMipmappedArray,
    ROCPROFILER_HIP_API_ID_hipMallocPitch,
    ROCPROFILER_HIP_API_ID_hipMemAddressFree,
    ROCPROFILER_HIP_API_ID_hipMemAddressReserve,
    ROCPROFILER_HIP_API_ID_hipMemAdvise,
    ROCPROFILER_HIP_API_ID_hipMemAllocHost,
    ROCPROFILER_HIP_API_ID_hipMemAllocPitch,
    ROCPROFILER_HIP_API_ID_hipMemCreate,
    ROCPROFILER_HIP_API_ID_hipMemExportToShareableHandle,
    ROCPROFILER_HIP_API_ID_hipMemGetAccess,
    ROCPROFILER_HIP_API_ID_hipMemGetAddressRange,
    ROCPROFILER_HIP_API_ID_hipMemGetAllocationGranularity,
    ROCPROFILER_HIP_API_ID_hipMemGetAllocationPropertiesFromHandle,
    ROCPROFILER_HIP_API_ID_hipMemGetInfo,
    ROCPROFILER_HIP_API_ID_hipMemImportFromShareableHandle,
    ROCPROFILER_HIP_API_ID_hipMemMap,
    ROCPROFILER_HIP_API_ID_hipMemMapArrayAsync,
    ROCPROFILER_HIP_API_ID_hipMemPoolCreate,
    ROCPROFILER_HIP_API_ID_hipMemPoolDestroy,
    ROCPROFILER_HIP_API_ID_hipMemPoolExportPointer,
    ROCPROFILER_HIP_API_ID_hipMemPoolExportToShareableHandle,
    ROCPROFILER_HIP_API_ID_hipMemPoolGetAccess,
    ROCPROFILER_HIP_API_ID_hipMemPoolGetAttribute,
    ROCPROFILER_HIP_API_ID_hipMemPoolImportFromShareableHandle,
    ROCPROFILER_HIP_API_ID_hipMemPoolImportPointer,
    ROCPROFILER_HIP_API_ID_hipMemPoolSetAccess,
    ROCPROFILER_HIP_API_ID_hipMemPoolSetAttribute,
    ROCPROFILER_HIP_API_ID_hipMemPoolTrimTo,
    ROCPROFILER_HIP_API_ID_hipMemPrefetchAsync,
    ROCPROFILER_HIP_API_ID_hipMemPtrGetInfo,
    ROCPROFILER_HIP_API_ID_hipMemRangeGetAttribute,
    ROCPROFILER_HIP_API_ID_hipMemRangeGetAttributes,
    ROCPROFILER_HIP_API_ID_hipMemRelease,
    ROCPROFILER_HIP_API_ID_hipMemRetainAllocationHandle,
    ROCPROFILER_HIP_API_ID_hipMemSetAccess,
    ROCPROFILER_HIP_API_ID_hipMemUnmap,
    ROCPROFILER_HIP_API_ID_hipMemcpy,
    ROCPROFILER_HIP_API_ID_hipMemcpy2D,
    ROCPROFILER_HIP_API_ID_hipMemcpy2DAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpy2DFromArray,
    ROCPROFILER_HIP_API_ID_hipMemcpy2DFromArrayAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpy2DToArray,
    ROCPROFILER_HIP_API_ID_hipMemcpy2DToArrayAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpy3D,
    ROCPROFILER_HIP_API_ID_hipMemcpy3DAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpyAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpyAtoH,
    ROCPROFILER_HIP_API_ID_hipMemcpyDtoD,
    ROCPROFILER_HIP_API_ID_hipMemcpyDtoDAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpyDtoH,
    ROCPROFILER_HIP_API_ID_hipMemcpyDtoHAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpyFromArray,
    ROCPROFILER_HIP_API_ID_hipMemcpyFromSymbol,
    ROCPROFILER_HIP_API_ID_hipMemcpyFromSymbolAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpyHtoA,
    ROCPROFILER_HIP_API_ID_hipMemcpyHtoD,
    ROCPROFILER_HIP_API_ID_hipMemcpyHtoDAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpyParam2D,
    ROCPROFILER_HIP_API_ID_hipMemcpyParam2DAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpyPeer,
    ROCPROFILER_HIP_API_ID_hipMemcpyPeerAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpyToArray,
    ROCPROFILER_HIP_API_ID_hipMemcpyToSymbol,
    ROCPROFILER_HIP_API_ID_hipMemcpyToSymbolAsync,
    ROCPROFILER_HIP_API_ID_hipMemcpyWithStream,
    ROCPROFILER_HIP_API_ID_hipMemset,
    ROCPROFILER_HIP_API_ID_hipMemset2D,
    ROCPROFILER_HIP_API_ID_hipMemset2DAsync,
    ROCPROFILER_HIP_API_ID_hipMemset3D,
    ROCPROFILER_HIP_API_ID_hipMemset3DAsync,
    ROCPROFILER_HIP_API_ID_hipMemsetAsync,
    ROCPROFILER_HIP_API_ID_hipMemsetD16,
    ROCPROFILER_HIP_API_ID_hipMemsetD16Async,
    ROCPROFILER_HIP_API_ID_hipMemsetD32,
    ROCPROFILER_HIP_API_ID_hipMemsetD32Async,
    ROCPROFILER_HIP_API_ID_hipMemsetD8,
    ROCPROFILER_HIP_API_ID_hipMemsetD8Async,
    ROCPROFILER_HIP_API_ID_hipMipmappedArrayCreate,
    ROCPROFILER_HIP_API_ID_hipMipmappedArrayDestroy,
    ROCPROFILER_HIP_API_ID_hipMipmappedArrayGetLevel,
    ROCPROFILER_HIP_API_ID_hipModuleGetFunction,
    ROCPROFILER_HIP_API_ID_hipModuleGetGlobal,
    ROCPROFILER_HIP_API_ID_hipModuleGetTexRef,
    ROCPROFILER_HIP_API_ID_hipModuleLaunchCooperativeKernel,
    ROCPROFILER_HIP_API_ID_hipModuleLaunchCooperativeKernelMultiDevice,
    ROCPROFILER_HIP_API_ID_hipModuleLaunchKernel,
    ROCPROFILER_HIP_API_ID_hipModuleLoad,
    ROCPROFILER_HIP_API_ID_hipModuleLoadData,
    ROCPROFILER_HIP_API_ID_hipModuleLoadDataEx,
    ROCPROFILER_HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor,
    ROCPROFILER_HIP_API_ID_hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
    ROCPROFILER_HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSize,
    ROCPROFILER_HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSizeWithFlags,
    ROCPROFILER_HIP_API_ID_hipModuleUnload,
    ROCPROFILER_HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessor,
    ROCPROFILER_HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
    ROCPROFILER_HIP_API_ID_hipOccupancyMaxPotentialBlockSize,
    ROCPROFILER_HIP_API_ID_hipPeekAtLastError,
    ROCPROFILER_HIP_API_ID_hipPointerGetAttribute,
    ROCPROFILER_HIP_API_ID_hipPointerGetAttributes,
    ROCPROFILER_HIP_API_ID_hipPointerSetAttribute,
    ROCPROFILER_HIP_API_ID_hipProfilerStart,
    ROCPROFILER_HIP_API_ID_hipProfilerStop,
    ROCPROFILER_HIP_API_ID_hipRuntimeGetVersion,
    ROCPROFILER_HIP_API_ID_hipSetDevice,
    ROCPROFILER_HIP_API_ID_hipSetDeviceFlags,
    ROCPROFILER_HIP_API_ID_hipSetupArgument,
    ROCPROFILER_HIP_API_ID_hipSignalExternalSemaphoresAsync,
    ROCPROFILER_HIP_API_ID_hipStreamAddCallback,
    ROCPROFILER_HIP_API_ID_hipStreamAttachMemAsync,
    ROCPROFILER_HIP_API_ID_hipStreamBeginCapture,
    ROCPROFILER_HIP_API_ID_hipStreamCreate,
    ROCPROFILER_HIP_API_ID_hipStreamCreateWithFlags,
    ROCPROFILER_HIP_API_ID_hipStreamCreateWithPriority,
    ROCPROFILER_HIP_API_ID_hipStreamDestroy,
    ROCPROFILER_HIP_API_ID_hipStreamEndCapture,
    ROCPROFILER_HIP_API_ID_hipStreamGetCaptureInfo,
    ROCPROFILER_HIP_API_ID_hipStreamGetCaptureInfo_v2,
    ROCPROFILER_HIP_API_ID_hipStreamGetDevice,
    ROCPROFILER_HIP_API_ID_hipStreamGetFlags,
    ROCPROFILER_HIP_API_ID_hipStreamGetPriority,
    ROCPROFILER_HIP_API_ID_hipStreamIsCapturing,
    ROCPROFILER_HIP_API_ID_hipStreamQuery,
    ROCPROFILER_HIP_API_ID_hipStreamSynchronize,
    ROCPROFILER_HIP_API_ID_hipStreamUpdateCaptureDependencies,
    ROCPROFILER_HIP_API_ID_hipStreamWaitEvent,
    ROCPROFILER_HIP_API_ID_hipStreamWaitValue32,
    ROCPROFILER_HIP_API_ID_hipStreamWaitValue64,
    ROCPROFILER_HIP_API_ID_hipStreamWriteValue32,
    ROCPROFILER_HIP_API_ID_hipStreamWriteValue64,
    ROCPROFILER_HIP_API_ID_hipTexObjectCreate,               // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipTexObjectDestroy,              // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipTexObjectGetResourceDesc,      // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipTexObjectGetResourceViewDesc,  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipTexObjectGetTextureDesc,       // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipTexRefGetAddress,
    ROCPROFILER_HIP_API_ID_hipTexRefGetAddressMode,  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipTexRefGetFilterMode,   // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipTexRefGetFlags,
    ROCPROFILER_HIP_API_ID_hipTexRefGetFormat,
    ROCPROFILER_HIP_API_ID_hipTexRefGetMaxAnisotropy,
    ROCPROFILER_HIP_API_ID_hipTexRefGetMipMappedArray,
    ROCPROFILER_HIP_API_ID_hipTexRefGetMipmapFilterMode,  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipTexRefGetMipmapLevelBias,
    ROCPROFILER_HIP_API_ID_hipTexRefGetMipmapLevelClamp,
    ROCPROFILER_HIP_API_ID_hipTexRefSetAddress,
    ROCPROFILER_HIP_API_ID_hipTexRefSetAddress2D,
    ROCPROFILER_HIP_API_ID_hipTexRefSetAddressMode,  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipTexRefSetArray,
    ROCPROFILER_HIP_API_ID_hipTexRefSetBorderColor,
    ROCPROFILER_HIP_API_ID_hipTexRefSetFilterMode,  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipTexRefSetFlags,
    ROCPROFILER_HIP_API_ID_hipTexRefSetFormat,
    ROCPROFILER_HIP_API_ID_hipTexRefSetMaxAnisotropy,
    ROCPROFILER_HIP_API_ID_hipTexRefSetMipmapFilterMode,  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipTexRefSetMipmapLevelBias,
    ROCPROFILER_HIP_API_ID_hipTexRefSetMipmapLevelClamp,
    ROCPROFILER_HIP_API_ID_hipTexRefSetMipmappedArray,
    ROCPROFILER_HIP_API_ID_hipThreadExchangeStreamCaptureMode,
    ROCPROFILER_HIP_API_ID_hipUnbindTexture,  // deprecated or removed
    ROCPROFILER_HIP_API_ID_hipUserObjectCreate,
    ROCPROFILER_HIP_API_ID_hipUserObjectRelease,
    ROCPROFILER_HIP_API_ID_hipUserObjectRetain,
    ROCPROFILER_HIP_API_ID_hipWaitExternalSemaphoresAsync,
    ROCPROFILER_HIP_API_ID_hipCreateChannelDesc,
    ROCPROFILER_HIP_API_ID_hipExtModuleLaunchKernel,
    ROCPROFILER_HIP_API_ID_hipHccModuleLaunchKernel,
    ROCPROFILER_HIP_API_ID_hipMemcpy_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpyToSymbol_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpyFromSymbol_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpy2D_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpy2DFromArray_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpy3D_spt,
    ROCPROFILER_HIP_API_ID_hipMemset_spt,
    ROCPROFILER_HIP_API_ID_hipMemsetAsync_spt,
    ROCPROFILER_HIP_API_ID_hipMemset2D_spt,
    ROCPROFILER_HIP_API_ID_hipMemset2DAsync_spt,
    ROCPROFILER_HIP_API_ID_hipMemset3DAsync_spt,
    ROCPROFILER_HIP_API_ID_hipMemset3D_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpyAsync_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpy3DAsync_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpy2DAsync_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpyFromSymbolAsync_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpyToSymbolAsync_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpyFromArray_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpy2DToArray_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpy2DFromArrayAsync_spt,
    ROCPROFILER_HIP_API_ID_hipMemcpy2DToArrayAsync_spt,
    ROCPROFILER_HIP_API_ID_hipStreamQuery_spt,
    ROCPROFILER_HIP_API_ID_hipStreamSynchronize_spt,
    ROCPROFILER_HIP_API_ID_hipStreamGetPriority_spt,
    ROCPROFILER_HIP_API_ID_hipStreamWaitEvent_spt,
    ROCPROFILER_HIP_API_ID_hipStreamGetFlags_spt,
    ROCPROFILER_HIP_API_ID_hipStreamAddCallback_spt,
    ROCPROFILER_HIP_API_ID_hipEventRecord_spt,
    ROCPROFILER_HIP_API_ID_hipLaunchCooperativeKernel_spt,
    ROCPROFILER_HIP_API_ID_hipLaunchKernel_spt,
    ROCPROFILER_HIP_API_ID_hipGraphLaunch_spt,
    ROCPROFILER_HIP_API_ID_hipStreamBeginCapture_spt,
    ROCPROFILER_HIP_API_ID_hipStreamEndCapture_spt,
    ROCPROFILER_HIP_API_ID_hipStreamIsCapturing_spt,
    ROCPROFILER_HIP_API_ID_hipStreamGetCaptureInfo_spt,
    ROCPROFILER_HIP_API_ID_hipStreamGetCaptureInfo_v2_spt,
    ROCPROFILER_HIP_API_ID_hipLaunchHostFunc_spt,
    ROCPROFILER_HIP_API_ID_hipGetStreamDeviceId,
    // ROCPROFILER_HIP_API_ID_hipDrvGraphAddMemsetNode,
    ROCPROFILER_HIP_API_ID_LAST,
} rocprofiler_hip_api_id_t;
