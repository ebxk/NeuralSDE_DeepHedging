"""Portable exact-array helpers; obsolete machine-specific replay CLI omitted."""
import hashlib
import json
import numpy as np

class EquivalenceError(AssertionError):
    def __init__(self,comparison):
        self.comparison=comparison
        super().__init__('full-array identity failed: '+json.dumps(comparison,ensure_ascii=False))

def _array_sha(array):
    array=np.ascontiguousarray(array)
    h=hashlib.sha256(str(array.dtype).encode()+repr(array.shape).encode())
    h.update(array.tobytes());return h.hexdigest()

def assert_full_array_identity(name,actual,cached):
    """Cast to the cached storage dtype, then require EVERY element exact.

    No sample, allclose, relative tolerance, financial-bound relaxation or
    policy-error exception can turn a nonexact replay into complete=True.
    """
    actual=np.asarray(actual);cached=np.asarray(cached)
    original_dtype=str(actual.dtype);actual=actual.astype(cached.dtype,copy=False)
    report=dict(name=name,actual_shape=list(actual.shape),cached_shape=list(cached.shape),
        original_actual_dtype=original_dtype,comparison_dtype=str(cached.dtype),
        compared_elements=int(cached.size),actual_sha256=_array_sha(actual),cached_sha256=_array_sha(cached),
        exact=False,different_elements=None,maximum_absolute_error=None,maximum_relative_error=None,
        first_difference_index=None)
    if actual.shape==cached.shape:
        different=actual!=cached;report['different_elements']=int(np.count_nonzero(different))
        finite=bool(np.isfinite(actual).all() and np.isfinite(cached).all())
        if finite:
            error=np.abs(actual.astype(float)-cached.astype(float))
            report['maximum_absolute_error']=float(np.max(error,initial=0))
            report['maximum_relative_error']=float(np.max(error/np.maximum(np.abs(cached.astype(float)),1e-30),initial=0))
        report['exact']=bool(finite and not np.any(different))
        if np.any(different):report['first_difference_index']=np.argwhere(different)[0].tolist()
    if not report['exact']:raise EquivalenceError(report)
    return report
