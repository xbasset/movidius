"""Read-only Apple Silicon Energy Model counters via IOReport; no sudo helper."""

import ctypes as C
from dataclasses import dataclass
import time


@dataclass
class Snapshot:
    pointer: int
    at: float


class MacEnergy:
    """Counter deltas are hardware-model estimates, not electrical meter readings."""

    def __init__(self):
        self.cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        self.io = C.CDLL("/usr/lib/libIOReport.dylib")
        p, u, n = C.c_void_p, C.c_uint64, C.c_long
        signatures = {
            "CFStringCreateWithCString": ([p, C.c_char_p, C.c_uint32], p),
            "CFStringGetCString": ([p, p, n, C.c_uint32], C.c_bool),
            "CFDictionaryGetValue": ([p, p], p),
            "CFDictionaryCreateMutableCopy": ([p, n, p], p),
            "CFDictionarySetValue": ([p, p, p], None),
            "CFArrayGetCount": ([p], n),
            "CFArrayGetValueAtIndex": ([p, n], p),
            "CFArrayCreateMutable": ([p, n, p], p),
            "CFArrayAppendValue": ([p, p], None),
            "CFRelease": ([p], None),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.cf, name)
            fn.argtypes, fn.restype = args, result
        signatures = {
            "IOReportCopyAllChannels": ([u, u], p),
            "IOReportCreateSubscription": ([p, p, C.POINTER(p), u, p], p),
            "IOReportCreateSamples": ([p, p, p], p),
            "IOReportCreateSamplesDelta": ([p, p, p], p),
            "IOReportChannelGetGroup": ([p], p),
            "IOReportChannelGetChannelName": ([p], p),
            "IOReportChannelGetUnitLabel": ([p], p),
            "IOReportSimpleGetIntegerValue": ([p, C.c_int], C.c_int64),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.io, name)
            fn.argtypes, fn.restype = args, result
        self.key = self.cf.CFStringCreateWithCString(None, b"IOReportChannels", 0x08000100)
        self.subscription, self.channels = None, None
        source = self.io.IOReportCopyAllChannels(0, 0)
        if not source:
            self.close()
            raise RuntimeError("IOReport ne fournit aucun compteur énergétique")
        selected = self.cf.CFArrayCreateMutable(
            None, 0, C.addressof(C.c_byte.in_dll(self.cf, "kCFTypeArrayCallBacks")))
        channels = self.cf.CFDictionaryCreateMutableCopy(None, 0, source)
        self.metadata = []
        try:
            for item in self.items(source):
                group, name, unit = self.describe(item)
                component = self.component(group, name)
                if component and unit in {"J", "mJ", "uJ", "µJ", "nJ"}:
                    self.cf.CFArrayAppendValue(selected, item)
                    self.metadata.append({"component": component, "channel": name, "unit": unit})
            if not self.metadata:
                raise RuntimeError("Compteurs Energy Model CPU/GPU indisponibles sur ce macOS")
            self.cf.CFDictionarySetValue(channels, self.key, selected)
            subscribed = p()
            self.subscription = self.io.IOReportCreateSubscription(None, channels, C.byref(subscribed), 0, None)
            self.channels = subscribed.value
            if not self.subscription or not self.channels:
                raise RuntimeError("Accès aux compteurs énergétiques refusé par macOS")
        except Exception:
            self.close()
            raise
        finally:
            for ref in [channels, selected, source]:
                self.cf.CFRelease(ref)

    def string(self, pointer):
        if not pointer:
            return ""
        buffer = C.create_string_buffer(512)
        if not self.cf.CFStringGetCString(pointer, buffer, len(buffer), 0x08000100):
            raise RuntimeError("Libellé IOReport illisible")
        return buffer.value.decode().strip()

    def items(self, sample):
        array = self.cf.CFDictionaryGetValue(sample, self.key)
        if array:
            for i in range(self.cf.CFArrayGetCount(array)):
                yield self.cf.CFArrayGetValueAtIndex(array, i)

    def describe(self, item):
        return tuple(self.string(fn(item)) for fn in [self.io.IOReportChannelGetGroup,
                     self.io.IOReportChannelGetChannelName, self.io.IOReportChannelGetUnitLabel])

    @staticmethod
    def component(group, name):
        if group != "Energy Model":
            return None
        if name.endswith("CPU Energy"):
            return "cpu"
        if name == "GPU Energy":
            return "gpu"
        if name.startswith("ANE"):
            return "ane"
        return None

    def snapshot(self):
        before = time.perf_counter()
        pointer = self.io.IOReportCreateSamples(self.subscription, self.channels, None)
        after = time.perf_counter()
        if not pointer:
            raise RuntimeError("Échantillon IOReport indisponible")
        return Snapshot(pointer, (before + after) / 2)

    def delta(self, before, after):
        difference = self.io.IOReportCreateSamplesDelta(before.pointer, after.pointer, None)
        if not difference:
            raise RuntimeError("Différence des compteurs IOReport indisponible")
        joules = {}
        try:
            for item in self.items(difference):
                group, name, unit = self.describe(item)
                component = self.component(group, name)
                factor = {"J": 1, "mJ": 1e-3, "uJ": 1e-6, "µJ": 1e-6, "nJ": 1e-9}.get(unit)
                if component and factor is not None:
                    value = self.io.IOReportSimpleGetIntegerValue(item, 0) * factor
                    if value < 0:
                        raise RuntimeError("Compteur énergétique réinitialisé pendant la mesure")
                    joules[component] = joules.get(component, 0) + value
        finally:
            self.cf.CFRelease(difference)
        seconds = after.at - before.at
        return {"seconds": seconds, "components_j": joules,
                "components_w": {k: v / seconds for k, v in joules.items()},
                "sum_j": sum(joules.values()), "sum_w": sum(joules.values()) / seconds,
                "source": "IOReport / Energy Model", "kind": "estimation matérielle",
                "scope": "Somme CPU + GPU + ANE du Mac disponibles ; ni bilan électrique complet ni consommation USB du stick"}

    def release(self, snapshot):
        self.cf.CFRelease(snapshot.pointer)

    def close(self):
        for name in ["subscription", "channels", "key"]:
            ref = getattr(self, name, None)
            if ref:
                self.cf.CFRelease(ref)
                setattr(self, name, None)
