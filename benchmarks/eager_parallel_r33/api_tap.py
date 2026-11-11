"""Read caller-thread counters from the explicitly preloaded diagnostic tap."""
import ctypes
from pathlib import Path


class Site(ctypes.Structure):
    _fields_ = [('caller', ctypes.c_uint64), ('count', ctypes.c_uint64),
                ('exclusive_ns', ctypes.c_uint64)]


class Bucket(ctypes.Structure):
    _fields_ = [('count', ctypes.c_uint64), ('inclusive_ns', ctypes.c_uint64),
                ('exclusive_ns', ctypes.c_uint64), ('caller', ctypes.c_uint64),
                ('overflow', ctypes.c_uint64), ('sites', Site*16)]


class ApiTap:
    def __init__(self, active):
        self.lib = ctypes.CDLL(None)
        self.lib.r33_phase_version.restype = ctypes.c_int
        assert self.lib.r33_phase_version() == 3
        self.lib.r33_phase_frequency.restype = ctypes.c_uint64
        self.frequency_hz = self.lib.r33_phase_frequency()
        self.lib.r33_phase_reset.argtypes = []
        self.lib.r33_phase_reset.restype = None
        self.lib.r33_phase_select.argtypes = [ctypes.c_int]
        self.lib.r33_phase_select.restype = None
        self.lib.r33_phase_tags.restype = ctypes.c_int
        self.lib.r33_phase_name.argtypes = [ctypes.c_int]
        self.lib.r33_phase_name.restype = ctypes.c_char_p
        self.lib.r33_phase_category.argtypes = [ctypes.c_int]
        self.lib.r33_phase_category.restype = ctypes.c_char_p
        self.lib.r33_phase_data.restype = ctypes.POINTER(Bucket)
        self.lib.r33_phase_next_lookups.argtypes = []
        self.lib.r33_phase_next_lookups.restype = ctypes.c_uint64
        self.tags = self.lib.r33_phase_tags()
        self.names = [self.lib.r33_phase_name(i).decode() for i in range(self.tags)]
        self.categories = [self.lib.r33_phase_category(i).decode() for i in range(self.tags)]
        self.active = active

    def reset(self):
        self.lib.r33_phase_reset()

    def phase(self, phase):
        self.lib.r33_phase_select(phase if self.active else -1)

    def snapshot(self):
        self.lib.r33_phase_select(-1)
        # Our task must not issue RTLD_NEXT lookups through the interposed
        # public dlsym, whose caller-sensitive semantics cannot be forwarded
        # by a generic C wrapper. Explicit CANN handles are unaffected.
        assert self.lib.r33_phase_next_lookups() == 0, 'RTLD_NEXT task is outside this tap contract'
        data = self.lib.r33_phase_data()
        maps = []
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split()
            if len(fields) >= 6 and fields[-1].startswith('/'):
                start, end = [int(x, 16) for x in fields[0].split('-')]
                maps.append((start, end, int(fields[2], 16), fields[-1]))

        def caller(address):
            for start, end, offset, path in maps:
                if start <= address < end:
                    return dict(address=address, object=path, offset=address-start+offset)
            return dict(address=address, object=None, offset=None)

        for i in range(2*self.tags):
            b = data[i]
            assert sum(s.count for s in b.sites)+b.overflow == b.count
            assert sum(s.exclusive_ns for s in b.sites) <= b.exclusive_ns
        return {phase: [dict(name=self.names[i], category=self.categories[i],
                            count=data[offset+i].count,
                            inclusive_ns=data[offset+i].inclusive_ns,
                            exclusive_ns=data[offset+i].exclusive_ns,
                            first_caller=data[offset+i].caller,
                            unattributed_site_calls=data[offset+i].overflow,
                            unattributed_site_ns=data[offset+i].exclusive_ns-
                                sum(s.exclusive_ns for s in data[offset+i].sites),
                            sites=[dict(**caller(s.caller), count=s.count, exclusive_ns=s.exclusive_ns)
                                   for s in data[offset+i].sites if s.count])
                        for i in range(self.tags)]
                for phase, offset in [('forward', 0), ('final_sync', self.tags)]}
