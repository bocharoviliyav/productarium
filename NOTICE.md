# NOTICE

This file carries the attribution notices for third-party software distributed
inside the Productarium docker image. Productarium itself is MIT-licensed —
see [LICENSE](LICENSE).

## Bundled database MCP servers

Productarium ships two third-party MCP servers so databases work out of the
box (the preset database flow, `api/mcp/presets.py`):

- **dbhub** — baked into the API image via `npm i -g @bytebase/dbhub`
  (the docker launcher fallback uses the `bytebase/dbhub` image).
- **oracle-mcp-server** — baked into the API image as a `uv` venv from the
  pinned git checkout `37ce2ead4e8caa274eb9442b44aff7f7a59573dd`
  (the docker launcher fallback uses the `dmeppiel/oracle-mcp-server` image).

Both are unmodified upstream distributions, launched as separate stdio
subprocesses at runtime.

---

## dbhub

- **Project:** <https://github.com/bytebase/dbhub>
- **Docker image:** `bytebase/dbhub`
- **License:** MIT — full text below.

```
MIT License

Copyright (c) 2025 Bytebase

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## oracle-mcp-server

- **Project:** <https://github.com/danielmeppiel/oracle-mcp-server>
- **Pinned commit:** `37ce2ead4e8caa274eb9442b44aff7f7a59573dd`
- **Docker image:** `dmeppiel/oracle-mcp-server`
- **License:** MIT — full text below.

```
MIT License

Copyright (c) 2025 MCP Oracle DB Context Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
