"""`.env.example` — контракт окружения. Проверяется паритет ИМЁН, а не наличие файла.

Порт из форка DeepWiki (ignore/deepwiki/tests/unit/test_env_example_parity.py),
адаптированный под продуктариум:

- имена берутся разбором AST по файлам ``api/`` НА ДИСКЕ (не по git-индексу:
  новые незакоммиченные модули подчиняются контракту так же, как трекаемые);
- образец документирует большинство переменных ЗАКОММЕНТИРОВАННЫМИ строками
  ``# NAME=value`` — они тоже считаются задокументированными;
- спец-источники, которых разбор вызовов не видит:
  * реестр таймаутов ``api/config/timeout.py`` — env-имена лежат в ДАННЫХ
    (``TimeoutKey(env_var=...)``), а не в вызовах ``os.*``;
  * ``${ENV_VAR}``-плейсхолдеры в ``api/config/*.json`` (читаются
    ``replace_env_placeholders`` во время загрузки конфигов);
  * ``process.env.*`` на фронте (``src/**`` + ``next.config.ts``).

Расхождение в любую сторону — это либо ненастраиваемая функция (код читает
переменную, которой нет в образце), либо мёртвая настройка (в образце есть
то, чего код не читает).
"""
import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_SECRET_SHAPES = (re.compile(r"^sk-"), re.compile(r"^AIza"), re.compile(r"^gh[pousr]_"),
                  re.compile(r"^glpat-"), re.compile(r"^xox"))


def _timeout_env_names() -> frozenset:
    try:
        from api.config.timeout import TIMEOUT_KEYS
        return frozenset(spec.env_var for spec in TIMEOUT_KEYS)
    except Exception:
        return frozenset()


def _api_python_files() -> list:
    """Файлы ``api/`` на диске: git-индекс отстаёт от рабочего дерева
    (новые незакоммиченные модули должны подчиняться контракту тоже)."""
    return sorted(p for p in (ROOT / "api").rglob("*.py")
                  if "__pycache__" not in p.parts)


def _names_from_config_placeholders() -> set:
    """``${ENV_VAR}`` в JSON-конфигах: читаются replace_env_placeholders."""
    names: set = set()
    for path in sorted((ROOT / "api" / "config").glob("*.json")):
        names.update(re.findall(r"\$\{([A-Z0-9_]+)\}",
                                path.read_text(encoding="utf-8", errors="ignore")))
    return names


def _module_string_constants(tree: ast.Module) -> dict:
    """Модульные строковые константы: ``PROBE_ENV = "DEEPWIKI_MODEL_WINDOW_PROBE"``.

    Без них детектор видит только литерал прямо в вызове, и ЛЮБАЯ переменная,
    вынесенная в константу, проходит мимо проверки.
    """
    constants: dict = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    return constants


def _names_read_by_code() -> set:
    names: set = set()
    for path in _api_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        constants = _module_string_constants(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                func = node.func
                is_getenv = (func.attr == "getenv" and isinstance(func.value, ast.Name)
                             and func.value.id == "os")
                # setdefault/pop по os.environ — тоже чтение контракта окружения.
                is_environ_call = (func.attr in {"get", "setdefault", "pop"}
                                   and isinstance(func.value, ast.Attribute)
                                   and func.value.attr == "environ")
                if (is_getenv or is_environ_call) and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        names.add(first.value)
                    elif isinstance(first, ast.Name) and first.id in constants:
                        names.add(constants[first.id])
            # Модульные хелперы ``_env_int``/``_env_float`` читают os.environ
            # внутри по ПАРАМЕТРУ — детектор видит только литерал на call-site.
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"_env_int", "_env_float"} and node.args):
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
            # Subscript в роли Load — чтение; Store (os.environ[...] = ...) —
            # запись write-through (OPENAI_API_KEY и др.), НЕ читаемый контракт.
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "environ"
                    and isinstance(node.ctx, ast.Load)):
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    names.add(node.slice.value)
                elif isinstance(node.slice, ast.Name) and node.slice.id in constants:
                    names.add(constants[node.slice.id])
    return names | _timeout_env_names() | _names_from_config_placeholders() | _names_read_by_frontend()


def _names_read_by_frontend() -> set:
    """``.env`` читает и фронт: next.config.ts и src/ берут из него адрес бэкенда."""
    names: set = set()
    files: list = [ROOT / "next.config.ts"]
    for pattern in ("*.ts", "*.tsx"):
        files.extend((ROOT / "src").rglob(pattern))
    for path in files:
        if path.is_file():
            names.update(re.findall(r"process\.env\.([A-Z_0-9]+)",
                                    path.read_text(encoding="utf-8", errors="ignore")))
    return names


_EXAMPLE_NAME = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=")


def _names_in_example() -> set:
    """Имена в образце, включая ЗАКОММЕНТИРОВАННЫЕ ``# NAME=value`` строки:
    так продуктариум документирует опциональные переменные."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return {m.group(1) for line in text.splitlines()
            if (m := _EXAMPLE_NAME.match(line))}


def test_every_variable_the_code_reads_is_documented():
    missing = _names_read_by_code() - _names_in_example()
    assert not missing, f"нет в .env.example: {sorted(missing)}"


def test_no_stale_variables_in_example():
    extra = _names_in_example() - _names_read_by_code()
    assert not extra, f"в .env.example есть то, чего трекаемый код не читает: {sorted(extra)}"


def test_example_reaches_a_clean_clone():
    """Образец ДОЙДЁТ до чистой машины: лежит в истории и не закрыт правилом."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", ".env.example"],
                             cwd=ROOT, capture_output=True, text=True)
    assert tracked.returncode == 0, ".env.example нет в истории — на чистой машине его не будет"

    rule = subprocess.run(["git", "check-ignore", "--no-index", "-v", ".env.example"],
                          cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if rule:
        pattern = rule.split("\t")[0].split(":")[-1]
        assert pattern.startswith("!"), (
            f".env.example закрыт правилом {pattern!r}: новый клон получит репозиторий "
            f"без образца окружения")


def test_example_carries_no_secrets():
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        assert not any(shape.match(value) for shape in _SECRET_SHAPES), f"секрет в {name}"
        if name.endswith("_API_KEY"):
            assert value in ("", "not-needed", "changeme"), f"{name} должен быть пустым в образце"
