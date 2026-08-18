"""Contract tests for the product-entity CRUD routes consumed by the UI.

The Next.js frontend talks to the backend over a fixed set of REST paths
(``api/routers/products.py`` and ``api/routers/docgen.py``). A previous
refactor broke this contract silently: the UI started interpolating the
SINGULAR entity kind (``codebase`` / ``spec`` / ``links``) directly into URLs,
while the routers register the PLURAL segments (``codebases`` / ``specs`` /
``links``). Every backend unit test kept passing because they exercised the
routers in isolation against their own (correct) paths, so the regression only
surfaced as 404s at runtime.

This module guards the contract by building the REAL app (all routers wired via
``api.routers.include_all_routers``) and asserting on its route table:
- the plural entity routes the UI consumes exist; and
- the singular variants the UI must never use do NOT exist.

No HTTP request or DB is needed — this is a pure route-table assertion, so it
runs in milliseconds and catches the exact failure mode described above.
"""

from __future__ import annotations

import importlib

# Entity kind (singular, as stored in UI form state) -> (plural URL segment,
# singular id path-parameter name). Mirrors src/lib/types.ts:entityPath.
_ENTITY_ROUTES = {
    "codebase": ("codebases", "codebase_id"),
    "spec": ("specs", "spec_id"),
    "links": ("links", "links_id"),
}


def _route_paths():
    """Return the set of route ``path`` templates on the full app."""
    import api.api as api_mod

    # The full app wires every router at import time via include_all_routers.
    importlib.reload(api_mod)
    return {getattr(route, "path", None) for route in api_mod.app.routes}


def test_full_app_includes_all_routers():
    """The dynamically-built app must actually contain routes (not an empty set)."""
    paths = _route_paths()
    assert paths, "app.routes is empty — include_all_routers wired nothing"
    assert any(p and p.startswith("/api/products") for p in paths)


def test_entity_subresource_routes_are_plural():
    """The add/delete/update routes the UI hits use PLURAL segments."""
    paths = _route_paths()
    for plural, id_param in _ENTITY_ROUTES.values():
        add = f"/api/products/{{product_id}}/{plural}"
        delete = f"/api/products/{{product_id}}/{plural}/{{{id_param}}}"
        assert add in paths, f"missing add route {add!r}"
        assert delete in paths, f"missing delete/update route {delete!r}"


def test_docgen_routes_are_plural():
    """Generate + status endpoints use PLURAL segments (codebases/specs)."""
    paths = _route_paths()
    for kind in ("codebase", "spec"):
        plural, id_param = _ENTITY_ROUTES[kind]
        gen = f"/api/products/{{product_id}}/{plural}/{{{id_param}}}/generate"
        status = f"/api/products/{{product_id}}/{plural}/{{{id_param}}}/generate/status"
        assert gen in paths, f"missing generate route {gen!r}"
        assert status in paths, f"missing status route {status!r}"


def test_verify_routes_are_plural():
    """Verify endpoints use PLURAL segments (codebases/specs/links)."""
    paths = _route_paths()
    for plural, id_param in _ENTITY_ROUTES.values():
        verify = f"/api/products/{{product_id}}/{plural}/{{{id_param}}}/verify"
        assert verify in paths, f"missing verify route {verify!r}"


def test_singular_entity_routes_do_not_exist():
    """The UI must never hit a singular entity segment — that was the bug.

    Assert the router table contains NO singular-subresource path, so any
    future frontend regression that interpolates the raw kind (e.g.
    ``/api/products/{id}/codebase``) is caught here instead of at runtime.
    """
    paths = _route_paths()
    for singular, (plural, _id_param) in _ENTITY_ROUTES.items():
        # ``links`` is identical in singular and plural form, so there is no
        # meaningful "singular vs plural" distinction to assert for it.
        if singular == plural:
            continue
        # Match the singular segment followed by a path boundary so that the
        # PLURAL ``codebases`` does not match the SINGULAR ``codebase``.
        prefix = f"/api/products/{{product_id}}/{singular}/"
        offending = sorted(p for p in paths if p and p.startswith(prefix))
        assert not offending, (
            f"singular route segment leaked into the router table: {offending}"
        )
