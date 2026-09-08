#!/usr/bin/env python3
"""Unit tests for api.llm.client.is_local_endpoint hardening (P0-10).

The old substring check treated ``http://localhost.attacker.com`` as local
(real-cred leak to an attacker domain) and missed IP/host.docker.internal
variants. The new check is an EXACT hostname match against the allowlist.
Ported from the former ``api/clients/openai_client.py`` client-method test
when the module was replaced by the langchain factory
(:func:`api.llm.client.is_local_endpoint`).
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from api.llm.client import is_local_endpoint as _is_local


class TestExactHostnameMatch:
    def test_localhost(self):
        assert _is_local("http://localhost:8080/v1") is True

    def test_localhost_no_port(self):
        assert _is_local("http://localhost/v1") is True

    def test_localhost_attacker_subdomain_not_local(self):
        """Regression: substring matching would have allowed this."""
        assert _is_local("http://localhost.attacker.com:8080/v1") is False

    def test_localhost_prefix_domain_not_local(self):
        assert _is_local("http://localhost-evil.com/v1") is False

    def test_attacker_host_with_localhost_path_not_local(self):
        assert _is_local("http://evil.com/localhost/v1") is False

    def test_loopback_ipv4(self):
        assert _is_local("http://127.0.0.1:1234/v1") is True

    def test_all_interfaces_ipv4(self):
        assert _is_local("http://0.0.0.0:1234/v1") is True

    def test_ipv6_loopback(self):
        assert _is_local("http://[::1]:1234/v1") is True

    def test_ipv6_all_interfaces(self):
        assert _is_local("http://[::]:1234/v1") is True

    def test_host_docker_internal(self):
        assert _is_local("http://host.docker.internal:8080/v1") is True

    def test_host_docker_internal_subdomain_not_local(self):
        assert _is_local("http://host.docker.internal.evil.io/v1") is False

    def test_remote_https(self):
        assert _is_local("https://api.openai.com/v1") is False

    def test_case_insensitive_host(self):
        assert _is_local("http://LOCALHOST:8080/v1") is True

    def test_empty_and_garbage_not_local(self):
        assert _is_local(None) is False
        assert _is_local("") is False
        assert _is_local("not a url") is False
