"""Public FNX2 logical contract and hardware-free reference runtime.

Physical adapters should consume these logical types and keep backend-native
buffer/layout details behind their adapter boundary.
"""

from .ragged_runtime import (
    IntegratedRaggedScheduler,
    RaggedBackendHealth,
    RaggedGenerationRequest,
    RaggedOrchestrator,
    RaggedReferenceStageBackend,
    RaggedRuntimeError,
    RaggedWorkerChannel,
    RaggedWorkerProcess,
    start_ragged_engine,
    stop_ragged_engine,
)
from .stage_abi_v2 import (
    ABI_MAJOR,
    ABI_MINOR,
    BatchDescriptor,
    LogicalTensor,
    LogicalTensorDescriptor,
    RaggedBatchRequest,
    RaggedBatchResult,
    RaggedFrame,
    RaggedFrameError,
    RaggedMessageKind,
    RaggedStageManifest,
    SequenceResult,
    SequenceSlice,
    decode_ragged_frame,
    derive_execution_lease_id,
    encode_ragged_frame,
    read_ragged_frame,
    send_ragged_frame,
)


FNX2_ABI_VERSION = (ABI_MAJOR, ABI_MINOR)

__all__ = [
    "BatchDescriptor",
    "FNX2_ABI_VERSION",
    "IntegratedRaggedScheduler",
    "LogicalTensor",
    "LogicalTensorDescriptor",
    "RaggedBackendHealth",
    "RaggedBatchRequest",
    "RaggedBatchResult",
    "RaggedFrame",
    "RaggedFrameError",
    "RaggedGenerationRequest",
    "RaggedMessageKind",
    "RaggedOrchestrator",
    "RaggedReferenceStageBackend",
    "RaggedRuntimeError",
    "RaggedStageManifest",
    "RaggedWorkerChannel",
    "RaggedWorkerProcess",
    "SequenceResult",
    "SequenceSlice",
    "decode_ragged_frame",
    "derive_execution_lease_id",
    "encode_ragged_frame",
    "read_ragged_frame",
    "send_ragged_frame",
    "start_ragged_engine",
    "stop_ragged_engine",
]
