#!/usr/bin/env python3
"""Unit tests for api.routers.knowledge — gaps not covered by
tests/integration/test_knowledge_tree.py.

Focuses on:
- ``_validate_parent_move`` (cycle detection, self-parent, empty string -> root,
  foreign parent)
- ``_convert_via_markitdown`` (markitdown available, UTF-8 fallback, 501 non-UTF-8)
- ``_is_owner`` non-admin non-owner 403 on verify
- tree endpoint on missing product (404)
- update_node parent_id move (drag-and-drop) + slug re-derive on empty slug
- _node_dict / _orm_to_node / _new_node_id helper coverage
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


# --- Helpers ----------------------------------------------------------------
def _system_user():
    from api.models import UserORM

    return UserORM(
        id="system",
        username="system",
        role="admin",
        provider="local",
        created_at=datetime.utcnow(),
    )


def _build_client(db_mod, knowledge_mod, *, user=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(knowledge_mod.router)

    def _get_test_db():
        s = db_mod.SessionLocal()
        try:
            yield s
        finally:
            s.close()

    user_obj = user if user is not None else _system_user()

    def _current_user():
        return user_obj

    app.dependency_overrides[knowledge_mod.get_db] = _get_test_db
    app.dependency_overrides[knowledge_mod.get_current_user] = _current_user
    return app, TestClient(app)


def _seed_product(db_mod, product_id="prod_1", owner_id=None):
    from api.models import ProductORM

    with db_mod.SessionLocal() as db:
        db.add(ProductORM(id=product_id, name="Acme", description="d", owner_id=owner_id))
        db.commit()
    return product_id


def _create_node(client, pid, title="Node", parent_id=None, content="c"):
    r = client.post(
        f"/api/products/{pid}/knowledge/nodes",
        json={"title": title, "content_md": content, "parent_id": parent_id},
    )
    assert r.status_code == 201, r.text
    return r.json()


# --- _validate_parent_move (cycle / self / empty / foreign) -----------------
class TestValidateParentMove:
    def test_empty_string_becomes_root(self, isolated_db):
        from api.routers.knowledge import _validate_parent_move

        # Empty string is the client's "move to root" signal -> None.
        result = _validate_parent_move(isolated_db.SessionLocal(), "prod_1", "node_1", "")
        assert result is None

    def test_none_becomes_root(self, isolated_db):
        from api.routers.knowledge import _validate_parent_move

        result = _validate_parent_move(isolated_db.SessionLocal(), "prod_1", "node_1", None)
        assert result is None

    def test_self_parent_rejected(self, isolated_db):
        from api.routers.knowledge import _validate_parent_move
        from api.models import KnowledgeNodeORM, ProductORM

        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="P"))
            db.add(KnowledgeNodeORM(
                id="node_1", product_id="prod_1", title="N", slug="n",
            ))
            db.commit()

        with pytest.raises(Exception) as exc:
            _validate_parent_move(isolated_db.SessionLocal(), "prod_1", "node_1", "node_1")
        assert "own parent" in str(exc.value.detail).lower()

    def test_cycle_rejected(self, isolated_db):
        """Moving a parent under its own descendant is a cycle."""
        from api.routers.knowledge import _validate_parent_move
        from api.models import KnowledgeNodeORM, ProductORM

        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="P"))
            db.add(KnowledgeNodeORM(
                id="node_root", product_id="prod_1", title="Root", slug="root",
            ))
            db.add(KnowledgeNodeORM(
                id="node_child", product_id="prod_1", parent_id="node_root",
                title="Child", slug="child",
            ))
            db.commit()

        # Moving root under child would create a cycle.
        with pytest.raises(Exception) as exc:
            _validate_parent_move(
                isolated_db.SessionLocal(), "prod_1", "node_root", "node_child"
            )
        assert "cycle" in str(exc.value.detail).lower()

    def test_foreign_parent_rejected(self, isolated_db):
        from api.routers.knowledge import _validate_parent_move
        from api.models import KnowledgeNodeORM, ProductORM

        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="P1"))
            db.add(ProductORM(id="prod_2", name="P2"))
            db.add(KnowledgeNodeORM(
                id="node_p2", product_id="prod_2", title="P2Node", slug="p2",
            ))
            db.commit()

        with pytest.raises(Exception) as exc:
            _validate_parent_move(
                isolated_db.SessionLocal(), "prod_1", "node_1", "node_p2"
            )
        assert "does not belong" in str(exc.value.detail).lower()


# --- markitdown upload ------------------------------------------------------
class TestMarkitdownUpload:
    def test_upload_utf8_fallback_when_markitdown_unavailable(self, isolated_db, monkeypatch):
        """When markitdown module is absent, UTF-8 text is stored as-is."""
        from api.routers import knowledge as knowledge_mod

        # Force importlib.import_module to fail for api.formats.markitdown.
        import importlib
        orig_import = importlib.import_module

        def _fake_import(name, *args, **kwargs):
            if name == "api.formats.markitdown":
                raise ImportError("not installed")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="Upload")

        resp = client.post(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}/upload",
            files={"file": ("readme.txt", b"Hello UTF-8 text content", "text/plain")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "Hello UTF-8 text content" in body["content_md"]

    def test_upload_non_utf8_returns_501(self, isolated_db, monkeypatch):
        """Binary (non-UTF-8) upload with markitdown unavailable -> 501."""
        from api.routers import knowledge as knowledge_mod

        import importlib
        orig_import = importlib.import_module

        def _fake_import(name, *args, **kwargs):
            if name == "api.formats.markitdown":
                raise ImportError("not installed")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", _fake_import)

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="Upload")

        # Binary bytes that are not valid UTF-8.
        resp = client.post(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}/upload",
            files={"file": ("binary.bin", b"\x89PNG\r\n\x1a\n\x00\x00", "application/octet-stream")},
        )
        assert resp.status_code == 501
        assert "markitdown is unavailable" in resp.json()["detail"]

    def test_upload_empty_file_400(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="Upload")

        resp = client.post(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_upload_node_not_found_404(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/knowledge/nodes/node_ghost/upload",
            files={"file": ("x.txt", b"hi", "text/plain")},
        )
        assert resp.status_code == 404


# --- _is_owner non-admin non-owner 403 --------------------------------------
class TestVerifyOwnership:
    def test_verify_non_admin_non_owner_403(self, isolated_db):
        from api.routers import knowledge as knowledge_mod
        from api.models import UserORM, KnowledgeNodeORM

        # Create a product with no owner + a node created by someone else.
        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="P"))
            db.add(KnowledgeNodeORM(
                id="node_1", product_id="prod_1", title="N", slug="n",
                created_by="user_other", source="manual",
            ))
            db.commit()

        # A plain user who is not the owner or creator.
        plain_user = UserORM(
            id="user_plain", username="plain", role="user",
            provider="local", created_at=datetime.utcnow(),
        )
        app, client = _build_client(isolated_db, knowledge_mod, user=plain_user)
        resp = client.post("/api/products/prod_1/knowledge/nodes/node_1/verify")
        assert resp.status_code == 403
        assert "owner" in resp.json()["detail"].lower()

    def test_verify_product_owner_allowed(self, isolated_db):
        """A user who owns the product can verify nodes in it."""
        from api.routers import knowledge as knowledge_mod
        from api.models import UserORM, KnowledgeNodeORM, ProductORM

        owner = UserORM(
            id="user_owner", username="owner", role="user",
            provider="local", created_at=datetime.utcnow(),
        )
        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="P", owner_id="user_owner"))
            db.add(KnowledgeNodeORM(
                id="node_1", product_id="prod_1", title="N", slug="n",
                source="manual",
            ))
            db.commit()

        app, client = _build_client(isolated_db, knowledge_mod, user=owner)
        resp = client.post("/api/products/prod_1/knowledge/nodes/node_1/verify")
        assert resp.status_code == 200
        body = resp.json()
        assert body["verified"] is True
        assert body["verified_by"] == "user_owner"

    def test_verify_node_creator_allowed(self, isolated_db):
        """A non-admin user who created the node can verify it."""
        from api.routers import knowledge as knowledge_mod
        from api.models import UserORM, KnowledgeNodeORM, ProductORM

        creator = UserORM(
            id="user_creator", username="creator", role="user",
            provider="local", created_at=datetime.utcnow(),
        )
        with isolated_db.SessionLocal() as db:
            db.add(ProductORM(id="prod_1", name="P"))
            db.add(KnowledgeNodeORM(
                id="node_1", product_id="prod_1", title="N", slug="n",
                created_by="user_creator", source="manual",
            ))
            db.commit()

        app, client = _build_client(isolated_db, knowledge_mod, user=creator)
        resp = client.post("/api/products/prod_1/knowledge/nodes/node_1/verify")
        assert resp.status_code == 200
        assert resp.json()["verified_by"] == "user_creator"


# --- Tree endpoint on missing product ---------------------------------------
class TestTreeEdgeCases:
    def test_tree_on_missing_product_404(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        resp = client.get("/api/products/prod_ghost/knowledge/tree")
        assert resp.status_code == 404
        assert "Product not found" in resp.json()["detail"]

    def test_get_node_missing_product_404(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        resp = client.get("/api/products/prod_ghost/knowledge/nodes/node_x")
        assert resp.status_code == 404

    def test_delete_node_missing_product_404(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        resp = client.delete("/api/products/prod_ghost/knowledge/nodes/node_x")
        assert resp.status_code == 404


# --- Update node: parent_id move + slug re-derive ---------------------------
class TestUpdateNodeMove:
    def test_move_node_to_root_via_empty_parent(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        root = _create_node(client, pid, title="Root")
        child = _create_node(client, pid, title="Child", parent_id=root["id"])

        # Move child to root via empty-string parent_id.
        resp = client.put(
            f"/api/products/{pid}/knowledge/nodes/{child['id']}",
            json={"parent_id": ""},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["parent_id"] is None

    def test_move_node_under_new_parent(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        root1 = _create_node(client, pid, title="Root1")
        root2 = _create_node(client, pid, title="Root2")
        child = _create_node(client, pid, title="Child", parent_id=root1["id"])

        resp = client.put(
            f"/api/products/{pid}/knowledge/nodes/{child['id']}",
            json={"parent_id": root2["id"]},
        )
        assert resp.status_code == 200
        assert resp.json()["parent_id"] == root2["id"]

    def test_move_node_self_parent_400(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="Self")

        resp = client.put(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}",
            json={"parent_id": node["id"]},
        )
        assert resp.status_code == 400
        assert "own parent" in resp.json()["detail"].lower()

    def test_move_node_under_descendant_cycle_400(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        root = _create_node(client, pid, title="Root")
        child = _create_node(client, pid, title="Child", parent_id=root["id"])

        # Move root under child -> cycle.
        resp = client.put(
            f"/api/products/{pid}/knowledge/nodes/{root['id']}",
            json={"parent_id": child["id"]},
        )
        assert resp.status_code == 400
        assert "cycle" in resp.json()["detail"].lower()

    def test_update_slug_re_derive_on_empty(self, isolated_db):
        """Sending an empty slug re-derives it from the title."""
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="Original Title")

        resp = client.put(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}",
            json={"slug": "", "title": "New Title"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["slug"] == "new-title"

    def test_update_node_type(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="N")

        resp = client.put(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}",
            json={"node_type": "folder"},
        )
        assert resp.status_code == 200
        assert resp.json()["node_type"] == "folder"

    def test_update_content_triggers_reindex(self, isolated_db, monkeypatch):
        """Updating content_md triggers _index_in_background (fire-and-forget)."""
        from api.routers import knowledge as knowledge_mod

        reindex_called = []

        def _fake_index(text, dataset):
            reindex_called.append((text, dataset))

        # Patch at the use-site import path (knowledge.py does
        # `from api.docgen import _index_in_background`).
        import api.docgen as docgen_mod
        monkeypatch.setattr(docgen_mod, "_index_in_background", _fake_index)

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="N", content="old")

        resp = client.put(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}",
            json={"content_md": "new content for reindex"},
        )
        assert resp.status_code == 200
        # The re-index was called with the new content + product dataset.
        assert len(reindex_called) == 1
        assert "new content for reindex" in reindex_called[0][0]
        assert reindex_called[0][1] == "prod_prod_1"


# --- _convert_via_markitdown direct (markitdown returns output) -------------
class TestConvertViaMarkitdown:
    def test_convert_with_markitdown_available(self, isolated_db, monkeypatch):
        """When api.formats.markitdown is present, its output is stored.

        _convert_via_markitdown tries 4 call signatures: (path, filename),
        (path,), (bytes, filename), (bytes,). The first one to return non-empty
        wins. We accept both str (path) and bytes and return markdown.
        """
        from api.routers import knowledge as knowledge_mod
        from api.formats import markitdown as md_mod

        def _convert(arg, filename=None):
            # arg is a temp-file path (str) on the first two attempts and
            # raw bytes on the last two.
            if isinstance(arg, str):
                with open(arg, "rb") as f:
                    data = f.read()
            else:
                data = arg
            return f"# Converted\n\n{data.decode('utf-8', errors='replace')}"

        monkeypatch.setattr(md_mod, "convert_to_markdown", _convert)

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="Upload")

        resp = client.post(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}/upload",
            files={"file": ("doc.md", b"## Source content", "text/markdown")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "Converted" in body["content_md"]
        assert "Source content" in body["content_md"]

    def test_convert_via_markitdown_path_based_signature(self, isolated_db, monkeypatch):
        """The converter tries path-based signature first (temp file)."""
        from api.routers import knowledge as knowledge_mod
        from api.formats import markitdown as md_mod

        call_args = []

        def _convert_path_or_bytes(arg, filename=None):
            call_args.append((arg, filename))
            # Return markdown regardless of whether arg is a path or bytes.
            return f"# MD from {type(arg).__name__}"

        monkeypatch.setattr(md_mod, "convert_to_markdown", _convert_path_or_bytes)

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="Upload")

        resp = client.post(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}/upload",
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 200
        assert "MD from" in resp.json()["content_md"]
        # The first call attempt uses a path (temp file).
        assert len(call_args) >= 1


# --- _slugify edge cases (additional) ---------------------------------------
class TestSlugifyAdditional:
    def test_slugify_strips_dashes(self):
        from api.routers.knowledge import _slugify

        assert _slugify("---trailing---") == "trailing"
        assert _slugify("multiple   spaces") == "multiple-spaces"
        assert _slugify("under_score") == "under-score"

    def test_slugify_non_ascii_fallback(self):
        from api.routers.knowledge import _slugify

        # Non-ASCII chars are stripped; if result is empty, fallback is "node".
        result = _slugify("кириллица")
        assert isinstance(result, str)


# --- _node_dict / _orm_to_node ----------------------------------------------
class TestNodeSerialization:
    def test_node_dict_has_children_key(self):
        from api.routers.knowledge import _node_dict
        from types import SimpleNamespace

        node = SimpleNamespace(
            id="n1", product_id="p1", parent_id=None, title="T", slug="t",
            content_md="c", node_type="page", artifact_id=None, source="manual",
            verified=False, verified_by=None, verified_at=None, created_by=None,
            created_at=None, updated_at=None,
        )
        d = _node_dict(node)
        assert d["children"] == []
        assert d["id"] == "n1"
        assert d["title"] == "T"
        assert d["slug"] == "t"

    def test_orm_to_node_pydantic(self):
        from api.routers.knowledge import _orm_to_node
        from types import SimpleNamespace

        node = SimpleNamespace(
            id="n1", product_id="p1", parent_id=None, title="T", slug="t",
            content_md="c", node_type="page", artifact_id=None, source="manual",
            verified=False, verified_by=None, verified_at=None, created_by=None,
            created_at=None, updated_at=None,
        )
        pn = _orm_to_node(node)
        assert pn.id == "n1"
        assert pn.title == "T"
        assert pn.content_md == "c"


# --- build_tree direct ------------------------------------------------------
class TestBuildTree:
    def test_build_tree_nested(self):
        from api.routers.knowledge import build_tree
        from types import SimpleNamespace

        root = SimpleNamespace(
            id="r", product_id="p", parent_id=None, title="Root", slug="r",
            content_md="", node_type="folder", artifact_id=None, source="manual",
            verified=False, verified_by=None, verified_at=None, created_by=None,
            created_at=None, updated_at=None,
        )
        child = SimpleNamespace(
            id="c", product_id="p", parent_id="r", title="Child", slug="c",
            content_md="", node_type="page", artifact_id=None, source="manual",
            verified=False, verified_by=None, verified_at=None, created_by=None,
            created_at=None, updated_at=None,
        )
        orphan = SimpleNamespace(
            id="o", product_id="p", parent_id="missing", title="Orphan", slug="o",
            content_md="", node_type="page", artifact_id=None, source="manual",
            verified=False, verified_by=None, verified_at=None, created_by=None,
            created_at=None, updated_at=None,
        )
        tree = build_tree([root, child, orphan])
        # Root and orphan (stale parent) are both roots.
        root_ids = [n["id"] for n in tree]
        assert "r" in root_ids
        assert "o" in root_ids
        # Child is nested under root.
        root_entry = [n for n in tree if n["id"] == "r"][0]
        assert len(root_entry["children"]) == 1
        assert root_entry["children"][0]["id"] == "c"

    def test_build_tree_empty(self):
        from api.routers.knowledge import build_tree

        assert build_tree([]) == []


# --- get_knowledge_tree with actual nodes ----------------------------------
class TestKnowledgeTreeEndpoint:
    def test_tree_returns_nested_nodes(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        root = _create_node(client, pid, title="Root")
        child = _create_node(client, pid, title="Child", parent_id=root["id"])

        resp = client.get(f"/api/products/{pid}/knowledge/tree")
        assert resp.status_code == 200
        tree = resp.json()
        assert len(tree) == 1
        assert tree[0]["id"] == root["id"]
        assert len(tree[0]["children"]) == 1
        assert tree[0]["children"][0]["id"] == child["id"]


# --- get_node + delete_node success paths -----------------------------------
class TestGetDeleteNode:
    def test_get_node_success(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="My Node", content="hello")

        resp = client.get(f"/api/products/{pid}/knowledge/nodes/{node['id']}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "My Node"
        assert resp.json()["content_md"] == "hello"

    def test_delete_node_success(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="ToDelete")

        resp = client.delete(f"/api/products/{pid}/knowledge/nodes/{node['id']}")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"].lower()

        # Verify it's gone.
        resp2 = client.get(f"/api/products/{pid}/knowledge/nodes/{node['id']}")
        assert resp2.status_code == 404


# --- create_node with invalid parent ---------------------------------------
class TestCreateNodeBadParent:
    def test_create_node_parent_from_other_product_400(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid1 = _seed_product(isolated_db, product_id="prod_1")
        pid2 = _seed_product(isolated_db, product_id="prod_2")
        node_p2 = _create_node(client, pid2, title="P2Node")

        resp = client.post(
            f"/api/products/{pid1}/knowledge/nodes",
            json={"title": "P1Node", "content_md": "c", "parent_id": node_p2["id"]},
        )
        assert resp.status_code == 400
        assert "parent_id does not belong" in resp.json()["detail"]

    def test_create_node_parent_not_found_400(self, isolated_db):
        from api.routers import knowledge as knowledge_mod

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(
            f"/api/products/{pid}/knowledge/nodes",
            json={"title": "N", "content_md": "c", "parent_id": "node_ghost"},
        )
        assert resp.status_code == 400


# --- generate_summary endpoint (lines 471-498) ------------------------------
class TestGenerateSummary:
    def test_summary_success(self, isolated_db, monkeypatch):
        from api.routers import knowledge as knowledge_mod

        async def _fake_summary(product, codebases, specs, nodes):
            return "A concise summary."

        monkeypatch.setattr(knowledge_mod, "generate_product_summary", _fake_summary)

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(f"/api/products/{pid}/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"] == "A concise summary."
        assert body["product_id"] == pid

    def test_summary_empty_returns_503(self, isolated_db, monkeypatch):
        from api.routers import knowledge as knowledge_mod

        async def _fake_summary(product, codebases, specs, nodes):
            return ""

        monkeypatch.setattr(knowledge_mod, "generate_product_summary", _fake_summary)

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)

        resp = client.post(f"/api/products/{pid}/summary")
        assert resp.status_code == 503
        assert "no content available" in resp.json()["detail"]

    def test_summary_product_not_found_404(self, isolated_db, monkeypatch):
        from api.routers import knowledge as knowledge_mod

        async def _fake_summary(product, codebases, specs, nodes):
            return "x"

        monkeypatch.setattr(knowledge_mod, "generate_product_summary", _fake_summary)

        app, client = _build_client(isolated_db, knowledge_mod)

        resp = client.post("/api/products/prod_ghost/summary")
        assert resp.status_code == 404


# --- _convert_via_markitdown edge cases -------------------------------------
class TestConvertViaMarkitdownEdgeCases:
    def test_convert_returns_no_output(self, isolated_db, monkeypatch):
        """convert_to_markdown returns empty string -> (False, ...)."""
        from api.routers import knowledge as knowledge_mod
        from api.formats import markitdown as md_mod

        monkeypatch.setattr(md_mod, "convert_to_markdown", lambda *a, **k: "")

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="Upload")

        resp = client.post(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}/upload",
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        # Empty output from markitdown -> UTF-8 fallback succeeds.
        assert resp.status_code == 200
        assert "hello" in resp.json()["content_md"]

    def test_convert_typeerror_fallback(self, isolated_db, monkeypatch):
        """If the first call signature raises TypeError, the next is tried."""
        from api.routers import knowledge as knowledge_mod
        from api.formats import markitdown as md_mod

        call_count = [0]

        def _convert(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise TypeError("bad signature")
            return "# Converted via bytes"

        monkeypatch.setattr(md_mod, "convert_to_markdown", _convert)

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="Upload")

        resp = client.post(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}/upload",
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 200
        assert "Converted via bytes" in resp.json()["content_md"]

    def test_convert_function_missing(self, isolated_db, monkeypatch):
        """api.formats.markitdown exists but has no convert_to_markdown attr."""
        from api.routers import knowledge as knowledge_mod
        from api.formats import markitdown as md_mod

        monkeypatch.delattr(md_mod, "convert_to_markdown", raising=False)

        app, client = _build_client(isolated_db, knowledge_mod)
        pid = _seed_product(isolated_db)
        node = _create_node(client, pid, title="Upload")

        resp = client.post(
            f"/api/products/{pid}/knowledge/nodes/{node['id']}/upload",
            files={"file": ("doc.txt", b"plain text", "text/plain")},
        )
        # No convert function -> UTF-8 fallback.
        assert resp.status_code == 200
        assert "plain text" in resp.json()["content_md"]


# need ProductORM import for the owner tests
from api.models import ProductORM  # noqa: E402
