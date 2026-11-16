"""R41 treatment qualification; all calls occur outside request timing."""
import ctypes
import os


def require(value, message):
    if not value:
        raise ValueError(message)


def expected_treatment(config, environment):
    treatment = config.get('r41_treatment')
    require(type(treatment) is dict and set(treatment) == {'event_wait', 'workspace_cache'},
            'explicit R41 treatment required')
    require(treatment['event_wait'] in ('spin', 'condition') and
            type(treatment['workspace_cache']) is int and treatment['workspace_cache'] in (0, 1),
            'invalid R41 treatment')
    require(environment.get('TORCH_NPU_EVENT_HOST_WAIT') == treatment['event_wait'] and
            environment.get('TORCH_NPU_WORKSPACE_CACHE') == str(treatment['workspace_cache']),
            'R41 environment differs from configured treatment')
    digest = config.get('native_sha256')
    require(type(digest) is str and len(digest) == 64 and
            all(c in '0123456789abcdef' for c in digest), 'explicit native SHA256 required')
    return dict(treatment)


class NativeTreatment:
    def __init__(self, library, config, environment):
        self.expected = expected_treatment(config, environment)
        self.library = ctypes.CDLL(str(library), mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
        self.mode = self.library.torch_npu_event_host_wait_mode
        self.mode.argtypes = []
        self.mode.restype = ctypes.c_char_p
        self.stats = self.library.OpApiWorkspaceCacheStatsR41
        self.stats.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.stats.restype = None

    def read(self):
        mode = self.mode().decode('ascii')
        require(mode == self.expected['event_wait'], 'effective native Event mode differs')
        return dict(event_wait=mode, workspace_cache=self.counters()['enabled'])

    def counters(self):
        values = (ctypes.c_uint64 * 8)()
        self.stats(values, len(values))
        result = dict(zip(('enabled', 'hits', 'allocations', 'growths', 'evictions',
                           'bypasses', 'retained_bytes', 'entries'), map(int, values)))
        require(result['enabled'] == self.expected['workspace_cache'],
                'effective native workspace mode differs')
        require(result['entries'] <= 4 and result['retained_bytes'] <= 64 * 1024 * 1024,
                'workspace retained cache exceeds declared bounds')
        return result


def audit_callers(rows, callers, enabled):
    require(type(rows) is list and len(rows) == callers and
            all(type(row) is dict for row in rows), 'caller TLS cache receipts missing')
    require(all(row['enabled'] == enabled for row in rows), 'caller treatment differs')
    if enabled:
        require(all(row['hits'] > 0 and row['allocations'] > 0 for row in rows),
                'workspace treatment did not reach every caller; no performance claim')
    else:
        require(all(not any(row[k] for k in ('hits', 'allocations', 'growths', 'evictions',
                                            'bypasses', 'retained_bytes', 'entries')) for row in rows),
                'disabled workspace cache unexpectedly active')
    return dict(passed=True, every_caller_reached=bool(enabled),
                observations_include_warmup=True, observations_outside_request_timing=True)
