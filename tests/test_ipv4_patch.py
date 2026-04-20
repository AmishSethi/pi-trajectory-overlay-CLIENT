"""Tests for the IPv4-only DNS patch in trajectory_predictor.py.

The patch filters IPv6 results out of socket.getaddrinfo so Python HTTP clients
(google-genai gRPC, openai, httpx) never try an IPv6 endpoint — fixes the
UPenn SEAS "SYN_SENT hangs for 60-90 s" issue.
"""
import importlib
import os
import socket
import sys

import pytest

REPO = os.path.expanduser("~/pi-trajectory-overlay")
sys.path.insert(0, REPO)


def _fresh_import():
    """Force-reimport trajectory_predictor so module-level patches re-run."""
    if "trajectory_predictor" in sys.modules:
        del sys.modules["trajectory_predictor"]
    import trajectory_predictor  # noqa: E402
    return trajectory_predictor


class TestPatchInstalled:
    """Module import should replace socket.getaddrinfo with the IPv4-only version."""

    def test_socket_getaddrinfo_is_patched(self, monkeypatch):
        monkeypatch.delenv("ALLOW_IPV6", raising=False)
        tp = _fresh_import()
        assert socket.getaddrinfo is tp._ipv4_only_getaddrinfo, (
            "socket.getaddrinfo was not replaced at import time"
        )

    def test_original_getaddrinfo_preserved(self):
        tp = _fresh_import()
        assert callable(tp._original_getaddrinfo)
        assert tp._original_getaddrinfo is not tp._ipv4_only_getaddrinfo


class TestFilteringLogic:
    """_ipv4_only_getaddrinfo itself filters out AF_INET6 results."""

    def test_filters_out_ipv6_even_if_returned(self, monkeypatch):
        tp = _fresh_import()
        fake_mixed = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::1", 443, 0, 0)),
            (socket.AF_INET,  socket.SOCK_STREAM, 6, "", ("1.2.3.4", 443)),
            (socket.AF_INET6, socket.SOCK_DGRAM,  17, "", ("::1", 443, 0, 0)),
            (socket.AF_INET,  socket.SOCK_DGRAM,  17, "", ("5.6.7.8", 443)),
        ]
        monkeypatch.setattr(tp, "_original_getaddrinfo", lambda *a, **kw: fake_mixed)
        out = tp._ipv4_only_getaddrinfo("example.com", 443)
        assert out, "should return at least one result"
        assert all(r[0] == socket.AF_INET for r in out), f"got mixed families: {out}"
        assert len(out) == 2

    def test_forces_family_af_inet_at_call_site(self, monkeypatch):
        tp = _fresh_import()
        captured = {}

        def _capture(host, port, family=0, type=0, proto=0, flags=0):
            captured["family"] = family
            return [(socket.AF_INET, 1, 6, "", ("1.2.3.4", int(port)))]

        monkeypatch.setattr(tp, "_original_getaddrinfo", _capture)
        # Caller asks for AF_UNSPEC (0), but the patch should force AF_INET.
        tp._ipv4_only_getaddrinfo("example.com", 443, family=socket.AF_UNSPEC)
        assert captured["family"] == socket.AF_INET

    def test_returns_empty_when_only_v6(self, monkeypatch):
        tp = _fresh_import()
        fake_only_v6 = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::1", 443, 0, 0)),
        ]
        monkeypatch.setattr(tp, "_original_getaddrinfo", lambda *a, **kw: fake_only_v6)
        out = tp._ipv4_only_getaddrinfo("v6only.example.com", 443)
        assert out == []


class TestBypassEnvVar:
    """ALLOW_IPV6=1 in the environment should skip installing the patch."""

    def test_allow_ipv6_does_not_patch(self, monkeypatch):
        monkeypatch.setenv("ALLOW_IPV6", "1")
        # Capture the real getaddrinfo BEFORE the import so we know what to expect.
        real = socket.getaddrinfo
        # If an earlier test already patched socket.getaddrinfo, unpatch first.
        if getattr(real, "__name__", "") == "_ipv4_only_getaddrinfo":
            # Force-reset using the known original captured inside the module.
            import trajectory_predictor as tp_prev
            socket.getaddrinfo = tp_prev._original_getaddrinfo
            real = socket.getaddrinfo
        tp = _fresh_import()
        assert socket.getaddrinfo is not tp._ipv4_only_getaddrinfo
