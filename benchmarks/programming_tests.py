#!/usr/bin/env python3
from __future__ import annotations
import ast
import builtins
import math


def _exec_code(code: str):
    glb = {"__builtins__": builtins, "math": math}
    loc = {}
    compiled = compile(code, filename="<patched>", mode="exec")
    exec(compiled, glb, loc)
    glb.update(loc)
    return glb


def test_mutable_default(code: str) -> bool:
    ns = _exec_code(code)
    fn = ns.get("concatenate")
    if not callable(fn):
        return False
    a = fn(1)
    b = fn(2)
    if a == b:
        return False
    c = fn(3, ['x'])
    return a == [1] and b == [2] and c == ['x', 3]


def test_off_by_one(code: str) -> bool:
    ns = _exec_code(code)
    fn = ns.get("inclusive_range")
    if not callable(fn):
        return False
    return fn(3) == [1, 2, 3] and fn(1) == [1]


def test_file_leak(code: str) -> bool:
    tree = ast.parse(code)
    has_with_open = False
    class V(ast.NodeVisitor):
        def visit_With(self, node: ast.With):
            nonlocal has_with_open
            for item in node.items:
                if isinstance(item.context_expr, ast.Call) and getattr(getattr(item.context_expr.func, 'id', None), 'lower', lambda: '' )().lower() == 'open':
                    has_with_open = True
            self.generic_visit(node)
    V().visit(tree)
    return has_with_open


def test_bare_except(code: str) -> bool:
    tree = ast.parse(code)
    class V(ast.NodeVisitor):
        ok = True
        def visit_Try(self, node: ast.Try):
            for h in node.handlers:
                if h.type is None:
                    V.ok = False
            self.generic_visit(node)
    v = V(); v.visit(tree)
    return v.ok


def test_shallow_copy(code: str) -> bool:
    ns = _exec_code(code)
    fn = ns.get("duplicate_and_modify")
    if not callable(fn):
        return False
    d = {'y': 0}
    out = fn(d)
    return 'x' not in d and isinstance(out, dict) and out.get('x') == 1


def test_sql_injection(code: str) -> bool:
    tree = ast.parse(code)
    class V(ast.NodeVisitor):
        parameterized = False
        def visit_Call(self, node: ast.Call):
            name = None
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name == 'execute' and len(node.args) >= 2:
                V.parameterized = True
            self.generic_visit(node)
    v = V(); v.visit(tree)
    return v.parameterized


def test_path_traversal(code: str) -> bool:
    return ('resolve(' in code) and ('startswith' in code)


def test_float_eq(code: str) -> bool:
    ns = _exec_code(code)
    fn = ns.get("is_one")
    if not callable(fn):
        return False
    a = fn(0.1 + 0.2)
    uses_isclose = 'isclose' in code
    return bool(a) and uses_isclose


TESTS = {
    'mutable_default': test_mutable_default,
    'off_by_one': test_off_by_one,
    'file_leak': test_file_leak,
    'bare_except': test_bare_except,
    'shallow_copy': test_shallow_copy,
    'sql_injection': test_sql_injection,
    'path_traversal': test_path_traversal,
    'float_eq': test_float_eq,
}


def run(case_id: str, code: str) -> bool:
    fn = TESTS.get(case_id)
    if not fn:
        return False
    try:
        return bool(fn(code))
    except Exception:
        return False

