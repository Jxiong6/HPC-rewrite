# ==============================================
# File: generate_cython_openmp_files.py (v3.3-pow2+finalsqrt+matmul-generic+elem2d)
# - 通用 GEMM（任意层序 + 三种写法）
# - 列表推导 / 一维写数组（单输出/多输出）
# - 归约（含最终 sqrt 检测）
# - 二维填充
# - ✅ 新增：二维逐元素（支持 A[i,j]/A[i,常数]/A[常数,j] + 一元数学函数）
# - ✅ 修复：int 常量不再变成 float（避免 3 -> 3.0 作为下标）
# - ✅ 修复：checks 缩进统一，避免 Inconsistent indentation
# ==============================================

from __future__ import annotations
import ast as _ast
from textwrap import dedent as _dedent

# -------------------------------- 公共头 --------------------------------
_HEADER = """# cython: boundscheck=False, wraparound=False, initializedcheck=False, nonecheck=False
# cython: cdivision=True, infer_types=True
from cython.parallel cimport prange, parallel
from openmp cimport omp_get_max_threads, omp_get_thread_num, omp_set_num_threads
from libc.math cimport sqrt, pow, fabs as c_fabs, exp as c_exp, log as c_log, sin as c_sin, cos as c_cos, tanh as c_tanh
cimport cython
import numpy as np
cimport numpy as np
"""

# ------------------------------ 基础排除集合（静态） ------------------------------
_DEF_EXCLUDES_STATIC = {
    # 常见循环索引
    "i", "j", "k", "l", "m", "idx",
    # Python 内置 / 迭代构造
    "len", "range",
    # 常见库与别名
    "np", "numpy", "math", "os", "sys", "time", "random",
    "pd", "sp", "torch", "tf", "plt",
    # 已导入的 Cython/OpenMP 名称（避免被当成参数）
    "cython", "omp_get_thread_num", "omp_get_max_threads", "omp_set_num_threads",
    # 数学函数名（由 cimport 提供）
    "sqrt",
}

# ------------------------------ 标识符工具 ------------------------------

def _idents_from_expr(expr_str: str, excludes: set[str]):
    try:
        node = _ast.parse(expr_str, mode="eval")
    except Exception:
        import re
        toks = re.findall(r"[A-Za-z_]\\w*", expr_str)
        out, seen = [], set()
        for t in toks:
            if t in excludes or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out
    out, seen = [], set()
    for n in _ast.walk(node):
        if isinstance(n, _ast.Name) and n.id not in excludes and n.id not in seen:
            seen.add(n.id)
            out.append(n.id)
    return out


def _params_from_sizes(excludes: set[str], *size_exprs: str):
    order, seen = [], set()
    for s in size_exprs:
        for name in _idents_from_expr(s, excludes):
            if name not in seen:
                seen.add(name)
                order.append(name)
    if not order:
        order = ["n"]
    params = ", ".join(f"int {v}" for v in order)
    return params, order

# ------------------------------ 表达式白名单 Writer ------------------------------
class _ExprWriter(_ast.NodeVisitor):
    def __init__(self, allowed_names: set[str], allowed_arrays: set[str]):
        self.allowed_names = allowed_names
        self.allowed_arrays = allowed_arrays

    def _chk(self, name):
        if name in self.allowed_names or name in self.allowed_arrays:
            return True
        raise ValueError(f"不允许的标识符: {name}")

    def visit_Name(self, node):
        self._chk(node.id)
        return node.id

    def visit_Constant(self, node):
        # ✅ 关键修复：保留 int 为整数文本，float 为浮点文本
        if isinstance(node.value, int):
            return repr(node.value)
        if isinstance(node.value, float):
            return repr(node.value)
        raise ValueError("仅允许数值常量")

    def visit_UnaryOp(self, node):
        s = self.visit(node.operand)
        if isinstance(node.op, _ast.UAdd):
            return f"+({s})"
        if isinstance(node.op, _ast.USub):
            return f"-({s})"
        raise ValueError("不支持的一元运算")

    def visit_BinOp(self, node):
        L = self.visit(node.left)
        R = self.visit(node.right)

        if isinstance(node.op, _ast.Pow):
            # ✅ 优先优化常见幂
            if isinstance(node.right, _ast.Constant):
                try:
                    rvf = float(node.right.value)
                except Exception:
                    raise ValueError("不支持的幂指数")

                # x ** 0.5 -> sqrt(x)
                if abs(rvf - 0.5) < 1e-12:
                    return f"sqrt({L})"

                # x ** n  (非负小整数) -> 乘法链
                if rvf.is_integer() and rvf >= 0 and rvf <= 12:
                    n = int(rvf)
                    if n == 0:
                        return "1.0"
                    if n == 1:
                        return f"({L})"
                    return "(" + " * ".join([f"({L})"] * n) + ")"

                # 其它常数幂：走 C pow
                return f"pow({L}, {rvf})"

            # ⭐ 非常数幂：走 C pow
            return f"pow({L}, {R})"

        # 其它二元运算
        op_map = {_ast.Add: "+", _ast.Sub: "-", _ast.Mult: "*", _ast.Div: "/"}
        for k, v in op_map.items():
            if isinstance(node.op, k):
                return f"({L} {v} {R})"
        raise ValueError("不支持的二元运算")

    def visit_IfExp(self, node):
        return f"({self.visit(node.body)} if {self.visit(node.test)} else {self.visit(node.orelse)})"

    def visit_Compare(self, node):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("仅支持简单比较")
        L, R = self.visit(node.left), self.visit(node.comparators[0])
        op = node.ops[0]
        op_map = {_ast.Lt: "<", _ast.LtE: "<=", _ast.Gt: ">", _ast.GtE: ">=", _ast.Eq: "==", _ast.NotEq: "!="}
        for k, v in op_map.items():
            if isinstance(op, k):
                return f"({L} {v} {R})"
        raise ValueError("不支持的比较运算")

    def visit_Call(self, node):
        def _one_arg(_):
            return len(node.args) == 1 and self.visit(node.args[0])

        # 允许的函数名（裸名）
        name_ok = isinstance(node.func, _ast.Name) and node.func.id in {
            "sqrt", "abs", "exp", "log", "sin", "cos", "tanh"
        }
        # 允许的 numpy 前缀：np.abs/np.sqrt 等
        np_ok = isinstance(node.func, _ast.Attribute) and isinstance(node.func.value, _ast.Name) \
                and node.func.value.id in {"np", "numpy"} and node.func.attr in {
                    "sqrt", "abs", "exp", "log", "sin", "cos", "tanh"
                }

        if name_ok or np_ok:
            arg = self.visit(node.args[0]) if _one_arg(node.args[0] if node.args else None) else None
            if arg is None:
                raise ValueError("仅允许一元数学函数调用")
            # 映射到 C 侧实现（abs -> fabs）
            if (name_ok and node.func.id == "abs") or (np_ok and node.func.attr == "abs"):
                return f"c_fabs({arg})"
            if (name_ok and node.func.id == "sqrt") or (np_ok and node.func.attr == "sqrt"):
                return f"sqrt({arg})"
            if (name_ok and node.func.id == "exp") or (np_ok and node.func.attr == "exp"):
                return f"c_exp({arg})"
            if (name_ok and node.func.id == "log") or (np_ok and node.func.attr == "log"):
                return f"c_log({arg})"
            if (name_ok and node.func.id == "sin") or (np_ok and node.func.attr == "sin"):
                return f"c_sin({arg})"
            if (name_ok and node.func.id == "cos") or (np_ok and node.func.attr == "cos"):
                return f"c_cos({arg})"
            if (name_ok and node.func.id == "tanh") or (np_ok and node.func.attr == "tanh"):
                return f"c_tanh({arg})"

        raise ValueError("仅允许受支持的一元数学函数调用")

    def visit_Subscript(self, node):
        tgt = node.value
        if not isinstance(tgt, _ast.Name) or tgt.id not in self.allowed_arrays:
            raise ValueError("仅允许在输入/输出数组上取下标")
        idx = node.slice
        if isinstance(idx, _ast.Tuple):
            if len(idx.elts) != 2:
                raise ValueError("仅允许二维下标")
            i0, i1 = self.visit(idx.elts[0]), self.visit(idx.elts[1])
            return f"{tgt.id}_mv[{i0}, {i1}]"
        else:
            i0 = self.visit(idx)
            return f"{tgt.id}_mv[{i0}]"

    def generic_visit(self, node):
        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

# ------------------------------- 语法分析器 -------------------------------
class _Analyzer:
    def __init__(self, src: str):
        self.src = _dedent(src).strip()
        self.mod = _ast.parse(self.src)
        self.fun = None
        for n in self.mod.body:
            if isinstance(n, _ast.FunctionDef):
                self.fun = n
                break
        self.root = _ast.Module(body=self.fun.body, type_ignores=[]) if self.fun else self.mod
        # 局部初始化（list/tuple/np.array/np.asarray）
        self.local_inits: dict[str, _ast.AST] = {}
        for stmt in (self.fun.body if self.fun else self.mod.body):
            if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], _ast.Name):
                nm = stmt.targets[0].id
                if isinstance(stmt.value, (_ast.List, _ast.Tuple)):
                    self.local_inits[nm] = stmt.value
                elif isinstance(stmt.value, _ast.Call) and isinstance(stmt.value.func, _ast.Attribute):
                    if isinstance(stmt.value.func.value, _ast.Name) and stmt.value.func.value.id in {"np", "numpy"} and stmt.value.func.attr in {"array", "asarray"}:
                        self.local_inits[nm] = stmt.value
        # 导入别名加入动态排除
        self.import_aliases: set[str] = set()
        for n in self.mod.body:
            if isinstance(n, _ast.Import):
                for a in n.names:
                    self.import_aliases.add(a.asname or a.name.split(".")[0])
            elif isinstance(n, _ast.ImportFrom):
                for a in n.names:
                    self.import_aliases.add(a.asname or a.name)

    def _walk(self):
        return _ast.walk(self.root)

    def list_comp(self):
        for n in self._walk():
            if isinstance(n, _ast.ListComp) and len(n.generators) == 1:
                gen = n.generators[0]
                if isinstance(gen.iter, _ast.Call) and isinstance(gen.iter.func, _ast.Name) and gen.iter.func.id == 'range':
                    return {"kind": "range", "comp": n, "gen": gen}
                if isinstance(gen.iter, _ast.Name):
                    return {"kind": "iterable", "comp": n, "gen": gen}
        return None

    def simple_for_assign(self):
        for n in self._walk():
            if isinstance(n, _ast.For) and isinstance(n.iter, _ast.Call) and isinstance(n.iter.func, _ast.Name) and n.iter.func.id == 'range':
                if len(n.body) == 1 and isinstance(n.body[0], _ast.Assign) and isinstance(n.body[0].targets[0], _ast.Subscript):
                    return {"for": n, "assign": n.body[0]}
        return None

    def simple_for_multi_assign(self):
        for n in self._walk():
            if isinstance(n, _ast.For) and isinstance(n.iter, _ast.Call) and isinstance(n.iter.func, _ast.Name) and n.iter.func.id == 'range':
                assigns, ok = [], True
                for stmt in n.body:
                    if isinstance(stmt, _ast.Assign) and isinstance(stmt.targets[0], _ast.Subscript):
                        assigns.append(stmt)
                    else:
                        ok = False
                        break
                if ok and len(assigns) >= 2:
                    return {"for": n, "assigns": assigns}
        return None

    def reduction_var(self):
        cand = None
        for n in self._walk():
            if isinstance(n, _ast.AugAssign) and isinstance(n.op, _ast.Add) and isinstance(n.target, _ast.Name) and n.target.id in {"result", "total"}:
                cand = n.target.id
        return cand

# ------------------------------ 2D 填充与工具 ------------------------------

def _base_name_of_2d_target(target_sub: _ast.Subscript):
    if isinstance(target_sub.slice, _ast.Tuple) and len(target_sub.slice.elts) == 2:
        return target_sub.value.id if isinstance(target_sub.value, _ast.Name) else None
    if isinstance(target_sub.value, _ast.Subscript):
        inner = target_sub.value
        return inner.value.id if isinstance(inner.value, _ast.Name) else None
    return None

def _detect_2d_fill_nested(root: _ast.AST):
    def _rhs_uses_only_indices(expr: _ast.AST, i_name: str, j_name: str) -> bool:
        for n in _ast.walk(expr):
            if isinstance(n, _ast.Name) and n.id not in {i_name, j_name}:
                return False
        return True

    for n in _ast.walk(root):
        if isinstance(n, _ast.For) and isinstance(n.iter, _ast.Call) and isinstance(n.iter.func, _ast.Name) and n.iter.func.id == 'range':
            i_name = n.target.id if isinstance(n.target, _ast.Name) else 'i'
            n0 = _ast.unparse(n.iter.args[0]) if n.iter.args else 'n'
            if not n.body:
                continue
            inner = n.body[0]
            if not (isinstance(inner, _ast.For) and isinstance(inner.iter, _ast.Call) and isinstance(inner.iter.func, _ast.Name) and inner.iter.func.id == 'range'):
                continue
            j_name = inner.target.id if isinstance(inner.target, _ast.Name) else 'j'
            n1 = _ast.unparse(inner.iter.args[0]) if inner.iter.args else 'n'
            for stmt in inner.body:
                if isinstance(stmt, _ast.Assign) and isinstance(stmt.targets[0], _ast.Subscript):
                    base = _base_name_of_2d_target(stmt.targets[0])
                    if base and _rhs_uses_only_indices(stmt.value, i_name, j_name):
                        return {"arr": base, "i": i_name, "j": j_name, "n0": n0, "n1": n1, "expr": stmt.value}
    return None

def _indices_only_ij_or_const(node: _ast.AST, i_name: str, j_name: str) -> bool:
    """
    判断 Subscript 的下标是否是 (i,j) / (i, 常数) / (常数, j) / (常数, 常数)。
    允许 a[i][j] 或 a[i,j] 两种形式。
    """
    base, i0, i1 = _extract_2d_indices(node) if isinstance(node, _ast.Subscript) else (None, None, None)
    if base is None:
        return False
    def _ok(x):
        return (isinstance(x, _ast.Name) and x.id in {i_name, j_name}) or isinstance(x, _ast.Constant)
    return _ok(i0) and _ok(i1)

def _collect_arrays_used_in_expr(expr: _ast.AST) -> set[str]:
    arrs = set()
    for n in _ast.walk(expr):
        if isinstance(n, _ast.Subscript) and isinstance(n.value, _ast.Name):
            arrs.add(n.value.id)
    return arrs

def _detect_2d_elementwise(root: _ast.AST):
    """
    通用二维逐元素：两层 for（任意顺序），内层有赋值：
        OUT[i,j] = <expr>
    expr 允许若干数组的 a[i,j]/a[i,常数]/a[常数,j]，并可与标量/一元数学函数组合。
    返回:
      {"out", "i", "j", "n0", "n1", "expr", "inputs"}
    """
    def _is_range_for(x):
        return isinstance(x, _ast.For) and isinstance(x.iter, _ast.Call) and \
               isinstance(x.iter.func, _ast.Name) and x.iter.func.id == 'range'

    for f1 in _ast.walk(root):
        if not _is_range_for(f1):
            continue
        i_name = _name_of_target(f1, "i")
        n0 = _ast.unparse(f1.iter.args[0]) if f1.iter.args else "n"
        for f2 in (s for s in f1.body if _is_range_for(s)):
            j_name = _name_of_target(f2, "j")
            n1 = _ast.unparse(f2.iter.args[0]) if f2.iter.args else "n"

            for stmt in f2.body:
                if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], _ast.Subscript):
                    out_base, oi0, oi1 = _extract_2d_indices(stmt.targets[0])
                    if out_base is None:
                        continue
                    if not (_nodes_equal(oi0, _ast.Name(id=i_name)) and _nodes_equal(oi1, _ast.Name(id=j_name))):
                        if _nodes_equal(oi0, _ast.Name(id=j_name)) and _nodes_equal(oi1, _ast.Name(id=i_name)):
                            i_name, j_name = j_name, i_name
                            n0, n1 = n1, n0
                        else:
                            continue

                    expr = stmt.value
                    ok = True
                    for n in _ast.walk(expr):
                        if isinstance(n, _ast.Subscript):
                            if not _indices_only_ij_or_const(n, i_name, j_name):
                                ok = False
                                break
                    if not ok:
                        continue

                    inputs = _collect_arrays_used_in_expr(expr)
                    if len(inputs) == 0:
                        inputs = set()

                    return {
                        "out": out_base, "i": i_name, "j": j_name,
                        "n0": n0, "n1": n1, "expr": expr,
                        "inputs": inputs
                    }
    return None


def _is_name(node, name):
    return isinstance(node, _ast.Name) and node.id == name

# --------- 通用下标解析：支持 a[i,j] 或 a[i][j] ----------
def _extract_2d_indices(sub: _ast.Subscript):
    # 直接 a[i,j]
    if isinstance(sub, _ast.Subscript) and isinstance(sub.value, _ast.Name):
        base = sub.value.id
        sl = sub.slice
        if isinstance(sl, _ast.Tuple) and len(sl.elts) == 2:
            return base, sl.elts[0], sl.elts[1]

    # 链式 a[i][j] / a[i][常数] / a[常数][j] / a[常数][常数]
    if isinstance(sub, _ast.Subscript) and isinstance(sub.value, _ast.Subscript):
        inner = sub.value
        if isinstance(inner.value, _ast.Name):
            base = inner.value.id
            i0 = inner.slice
            i1 = sub.slice
            if isinstance(i0, (_ast.Name, _ast.Constant)) and isinstance(i1, (_ast.Name, _ast.Constant)):
                return base, i0, i1

    return None, None, None


# -------------------------- 通用 GEMM 检测（任意循环层序 + 多种写法） --------------------------

def _collect_three_nested_fors(root: _ast.AST):
    def _is_range_for(x):
        return isinstance(x, _ast.For) and isinstance(x.iter, _ast.Call) and \
               isinstance(x.iter.func, _ast.Name) and x.iter.func.id == 'range'

    triples = []
    for f1 in _ast.walk(root):
        if not _is_range_for(f1):
            continue
        for f2 in (s for s in f1.body if _is_range_for(s)):
            for f3 in (s for s in f2.body if _is_range_for(s)):
                triples.append((f1, f2, f3))
    return triples


def _name_of_target(f: _ast.For, fallback: str) -> str:
    return f.target.id if isinstance(f.target, _ast.Name) else fallback

def _nodes_equal(a: _ast.AST, b: _ast.AST) -> bool:
    return isinstance(a, _ast.Name) and isinstance(b, _ast.Name) and a.id == b.id

def _match_A_B_C_pattern(A_sub: _ast.Subscript, B_sub: _ast.Subscript, C_sub: _ast.Subscript):
    Abase, Ai0, Ai1 = _extract_2d_indices(A_sub)
    Bbase, Bi0, Bi1 = _extract_2d_indices(B_sub)
    Cbase, Ci0, Ci1 = _extract_2d_indices(C_sub)
    if None in (Abase, Ai0, Ai1, Bbase, Bi0, Bi1, Cbase, Ci0, Ci1):
        return None
    if not _nodes_equal(Ai1, Bi0):
        return None
    k = Ai1
    i = Ai0
    j = Bi1
    if not (_nodes_equal(Ci0, i) and _nodes_equal(Ci1, j)):
        return None
    if not (isinstance(i, _ast.Name) and isinstance(j, _ast.Name) and isinstance(k, _ast.Name)):
        return None
    return (i.id, j.id, k.id, Abase, Bbase, Cbase)

def _find_acc_init_and_writeback(j_body: list[_ast.stmt], C_name: str, i_name: str, j_name: str):
    acc_name = None
    k_pos = None
    for idx, stmt in enumerate(j_body):
        if isinstance(stmt, _ast.For) and isinstance(stmt.iter, _ast.Call) and isinstance(stmt.iter.func, _ast.Name) and stmt.iter.func.id == 'range':
            k_pos = idx
            break
    if k_pos is None:
        return None

    for s in j_body[:k_pos]:
        if isinstance(s, _ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], _ast.Name):
            tgt = s.targets[0].id
            if isinstance(s.value, _ast.Constant) and s.value.value in (0, 0.0):
                acc_name = tgt

    if acc_name is None:
        return None

    for s in j_body[k_pos+1:]:
        if isinstance(s, _ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], _ast.Subscript):
            base, i0, i1 = _extract_2d_indices(s.targets[0])
            if base == C_name and isinstance(s.value, _ast.Name) and s.value.id == acc_name:
                if isinstance(i0, _ast.Name) and isinstance(i1, _ast.Name) and i0.id == i_name and i1.id == j_name:
                    return acc_name
    return None

def _detect_matmul_generic(root: _ast.AST):
    triples = _collect_three_nested_fors(root)
    for f1, f2, f3 in triples:
        v1 = _name_of_target(f1, "i")
        v2 = _name_of_target(f2, "j")
        v3 = _name_of_target(f3, "k")
        loop_vars = {v1, v2, v3}

        for stmt in f3.body:
            # 1) C[i,j] += A[i,k]*B[k,j]
            if isinstance(stmt, _ast.AugAssign) and isinstance(stmt.op, _ast.Add):
                if isinstance(stmt.target, _ast.Subscript) and isinstance(stmt.value, _ast.BinOp) and isinstance(stmt.value.op, _ast.Mult):
                    C_sub = stmt.target
                    A_sub = stmt.value.left
                    B_sub = stmt.value.right
                    if isinstance(A_sub, _ast.Subscript) and isinstance(B_sub, _ast.Subscript):
                        m = _match_A_B_C_pattern(A_sub, B_sub, C_sub)
                        if m:
                            iN, jN, kN, Aname, Bname, Cname = m
                            if {iN, jN, kN} == loop_vars:
                                return {"A": Aname, "B": Bname, "C": Cname, "i": iN, "j": jN, "k": kN}

            # 2) C[i,j] = C[i,j] + A[i,k]*B[k,j]
            if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], _ast.Subscript):
                C_sub_tgt = stmt.targets[0]
                val = stmt.value
                if isinstance(val, _ast.BinOp) and isinstance(val.op, _ast.Add):
                    if isinstance(val.left, _ast.Subscript):
                        C_sub_lhs = val.left
                        if _match_A_B_C_pattern_is_same_C(C_sub_tgt, C_sub_lhs):
                            if isinstance(val.right, _ast.BinOp) and isinstance(val.right.op, _ast.Mult):
                                A_sub = val.right.left
                                B_sub = val.right.right
                                if isinstance(A_sub, _ast.Subscript) and isinstance(B_sub, _ast.Subscript):
                                    m = _match_A_B_C_pattern(A_sub, B_sub, C_sub_tgt)
                                    if m:
                                        iN, jN, kN, Aname, Bname, Cname = m
                                        if {iN, jN, kN} == loop_vars:
                                            return {"A": Aname, "B": Bname, "C": Cname, "i": iN, "j": jN, "k": kN}

        # 3) 临时 acc
        for stmt in f3.body:
            if isinstance(stmt, _ast.AugAssign) and isinstance(stmt.op, _ast.Add):
                if isinstance(stmt.target, _ast.Name) and isinstance(stmt.value, _ast.BinOp) and isinstance(stmt.value.op, _ast.Mult):
                    acc = stmt.target.id
                    A_sub = stmt.value.left
                    B_sub = stmt.value.right
                    if isinstance(A_sub, _ast.Subscript) and isinstance(B_sub, _ast.Subscript):
                        for s2 in f2.body:
                            if isinstance(s2, _ast.Assign) and len(s2.targets) == 1 and isinstance(s2.targets[0], _ast.Subscript):
                                C_sub = s2.targets[0]
                                Cbase, Ci0, Ci1 = _extract_2d_indices(C_sub)
                                if Cbase and isinstance(s2.value, _ast.Name) and s2.value.id == acc:
                                    m = _match_A_B_C_pattern(A_sub, B_sub, C_sub)
                                    if m:
                                        iN, jN, kN, Aname, Bname, Cname = m
                                        if {iN, jN, kN} == loop_vars:
                                            acc_checked = _find_acc_init_and_writeback(f2.body, Cname, iN, jN)
                                            if acc_checked == acc:
                                                return {"A": Aname, "B": Bname, "C": Cname, "i": iN, "j": jN, "k": kN}

            if isinstance(stmt, _ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], _ast.Name):
                acc = stmt.targets[0].id
                val = stmt.value
                if isinstance(val, _ast.BinOp) and isinstance(val.op, _ast.Add):
                    if isinstance(val.left, _ast.Name) and val.left.id == acc and isinstance(val.right, _ast.BinOp) and isinstance(val.right.op, _ast.Mult):
                        A_sub = val.right.left
                        B_sub = val.right.right
                        if isinstance(A_sub, _ast.Subscript) and isinstance(B_sub, _ast.Subscript):
                            for s2 in f2.body:
                                if isinstance(s2, _ast.Assign) and len(s2.targets) == 1 and isinstance(s2.targets[0], _ast.Subscript):
                                    C_sub = s2.targets[0]
                                    m = _match_A_B_C_pattern(A_sub, B_sub, C_sub)
                                    if m:
                                        iN, jN, kN, Aname, Bname, Cname = m
                                        if {iN, jN, kN} == loop_vars:
                                            acc_checked = _find_acc_init_and_writeback(f2.body, Cname, iN, jN)
                                            if acc_checked == acc:
                                                return {"A": Aname, "B": Bname, "C": Cname, "i": iN, "j": jN, "k": kN}

    return None

def _match_A_B_C_pattern_is_same_C(C1: _ast.Subscript, C2: _ast.Subscript) -> bool:
    b1, i10, i11 = _extract_2d_indices(C1)
    b2, i20, i21 = _extract_2d_indices(C2)
    if None in (b1, i10, i11, b2, i20, i21):
        return False
    return (b1 == b2) and _nodes_equal(i10, i20) and _nodes_equal(i11, i21)

# ---------------- 嵌套 for/内联/if（其余功能保持不变） ----------------

def _collect_nested_for_indices(for_node: _ast.For):
    idx_names, ranges = [], []
    cur = for_node
    while isinstance(cur, _ast.For) and isinstance(cur.iter, _ast.Call) and isinstance(cur.iter.func, _ast.Name) and cur.iter.func.id == 'range':
        name = cur.target.id if isinstance(cur.target, _ast.Name) else 'i'
        rng = _ast.unparse(cur.iter.args[0]) if cur.iter.args else 'n'
        idx_names.append(name)
        ranges.append(rng)
        if cur.body and isinstance(cur.body[0], _ast.For):
            cur = cur.body[0]
        else:
            break
    return idx_names, ranges, cur

def _find_last_assignment_expr(root: _ast.AST, name: str) -> _ast.AST | None:
    last = None
    for node in _ast.walk(root):
        if isinstance(node, _ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], _ast.Name) and node.targets[0].id == name:
            last = node.value
    return last

def _find_enclosing_if_with_node(root: _ast.AST, target: _ast.AST) -> _ast.If | None:
    for node in _ast.walk(root):
        if isinstance(node, _ast.If):
            for sub in node.body:
                if target in set(_ast.walk(sub)):
                    return node
    return None

def _inline_simple_names(expr: _ast.AST, scope_root: _ast.AST, allowed_names: set[str], max_depth: int = 3) -> _ast.AST:
    if max_depth <= 0 or expr is None:
        return expr
    class Repl(_ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in allowed_names:
                return node
            rhs = _find_last_assignment_expr(scope_root, node.id)
            return rhs if rhs is not None else node
    new_expr = Repl().visit(expr)
    if new_expr is not expr:
        return _inline_simple_names(new_expr, scope_root, allowed_names, max_depth - 1)
    return new_expr

# 最终 sqrt 检测
def _need_final_sqrt(mod_or_fun_body: _ast.AST, var: str) -> bool:
    body = mod_or_fun_body.body if isinstance(mod_or_fun_body, _ast.FunctionDef) else (mod_or_fun_body.body if isinstance(mod_or_fun_body, _ast.Module) else [])
    last_ret = None
    for n in body:
        if isinstance(n, _ast.Return):
            last_ret = n
    if last_ret is None or last_ret.value is None:
        return False
    v = last_ret.value
    if isinstance(v, _ast.Call) and isinstance(v.func, _ast.Name) and v.func.id == 'sqrt' and len(v.args) == 1 and isinstance(v.args[0], _ast.Name) and v.args[0].id == var:
        return True
    if isinstance(v, _ast.BinOp) and isinstance(v.op, _ast.Pow) and isinstance(v.left, _ast.Name) and v.left.id == var and isinstance(v.right, _ast.Constant):
        try:
            return abs(float(v.right.value) - 0.5) < 1e-12
        except Exception:
            return False
    return False

# ------------------------------ 代码生成 ------------------------------

def _thread_default_stmt(default_threads):
    if default_threads is None:
        return ""
    return "if num_threads > 0: omp_set_num_threads(num_threads)\n    else: pass"

# ✅ 每段检查都自带 4 空格缩进，避免拼接时顶格
def _emit_checks_1d(name: str) -> str:
    return (
        f"    if {name}.dtype != np.float64 or not {name}.flags.c_contiguous:\n"
        f"        raise ValueError(\"{name} must be float64 and C-contiguous\")\n"
    )

def _emit_checks_2d(name: str) -> str:
    return (
        f"    if {name}.dtype != np.float64 or not {name}.flags.c_contiguous:\n"
        f"        raise ValueError(\"{name} must be float64 and C-contiguous\")\n"
    )

def _schedule_literal(schedule: str) -> str:
    return schedule if schedule in {"static", "dynamic", "guided"} else "static"

def _gen_matmul_gemm(func_name, A, B, C, N_expr, default_threads, schedule):
    sched = _schedule_literal(schedule)
    return f"""
{_HEADER}

def {func_name}(np.ndarray[np.double_t, ndim=2] {A},
                np.ndarray[np.double_t, ndim=2] {B},
                int num_threads=0):
    cdef Py_ssize_t N = {A}.shape[0]
    # 正确性检查
    if {A}.dtype != np.float64 or {B}.dtype != np.float64:
        raise ValueError("A, B must be float64")
    if not {A}.flags.c_contiguous or not {B}.flags.c_contiguous:
        raise ValueError("A, B must be C-contiguous")
    if {A}.shape[1] != N or {B}.shape[0] != N or {B}.shape[1] != N:
        raise ValueError("A, B must be square and compatible (N x N)")

    cdef np.ndarray[np.double_t, ndim=2] {C} = np.empty((N, N), dtype=np.float64)

    cdef const double[:, ::1] A_mv = {A}
    cdef const double[:, ::1] B_mv = {B}
    cdef double[:, ::1] C_mv = {C}

    cdef Py_ssize_t i, j

    if num_threads > 0:
        omp_set_num_threads(num_threads)

    with nogil, parallel():
        for i in prange(N, schedule="{sched}"):
            for j in range(N):
                C_mv[i, j] = _dot_rowcol(A_mv, B_mv, N, i, j)
    return {C}

cdef inline double _dot_rowcol(const double[:, ::1] A_mv,
                               const double[:, ::1] B_mv,
                               Py_ssize_t N, Py_ssize_t i, Py_ssize_t j) nogil:
    cdef Py_ssize_t k
    cdef double s = 0.0
    for k in range(N):
        s += A_mv[i, k] * B_mv[k, j]
    return s
"""

def _gen_matmul_gemm_mkn(func_name, A, B, C, M_expr, K_expr, N_expr, default_threads, schedule):
    sched = _schedule_literal(schedule)
    return f"""
{_HEADER}

def {func_name}(np.ndarray[np.double_t, ndim=2] {A},
                np.ndarray[np.double_t, ndim=2] {B},
                int num_threads=0):
    cdef Py_ssize_t M = {A}.shape[0]
    cdef Py_ssize_t K = {A}.shape[1]
    cdef Py_ssize_t N = {B}.shape[1]

    if {A}.dtype != np.float64 or {B}.dtype != np.float64:
        raise ValueError("A, B must be float64")
    if not {A}.flags.c_contiguous:
        raise ValueError("A must be C-contiguous")
    if {B}.shape[0] != K:
        raise ValueError("Shapes mismatch: A(M,K) x B(K,N)")

    cdef np.ndarray[np.double_t, ndim=2] BT = np.ascontiguousarray({B}.T)
    cdef np.ndarray[np.double_t, ndim=2] C = np.empty((M, N), dtype=np.float64)

    cdef const double[:, ::1] A_mv = {A}
    cdef const double[:, ::1] BT_mv = BT   # (N, K)
    cdef double[:, ::1] C_mv = C

    cdef Py_ssize_t i, j

    if num_threads > 0:
        omp_set_num_threads(num_threads)

    with nogil, parallel():
        for i in prange(M, schedule="{sched}"):
            for j in range(N):
                C_mv[i, j] = _dot_row_BTrow(A_mv, BT_mv, K, i, j)
    return C

cdef inline double _dot_row_BTrow(const double[:, ::1] A_mv,
                                  const double[:, ::1] BT_mv,
                                  Py_ssize_t K,
                                  Py_ssize_t i,
                                  Py_ssize_t j) nogil:
    cdef Py_ssize_t k
    cdef double s = 0.0
    for k in range(K):
        s += A_mv[i, k] * BT_mv[j, k]
    return s
"""

# ---- 列表推导（iterable） ----
def _gen_list_comp_iterable(func_name, a_name: str, iter_var: str, elt_node: _ast.AST, local_init: _ast.AST | None, default_threads, schedule):
    class Repl(_ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == iter_var:
                return _ast.Subscript(value=_ast.Name(id=a_name, ctx=_ast.Load()), slice=_ast.Name(id='i', ctx=_ast.Load()), ctx=_ast.Load())
            return node
    elt2 = Repl().visit(elt_node)

    allowed_names = {"i", "n"}
    writer = _ExprWriter(allowed_names=allowed_names, allowed_arrays={a_name})
    expr_c = writer.visit(elt2)
    sched = _schedule_literal(schedule)
    THRESH = 50000

    if local_init is None:
        sig = f"np.ndarray[np.double_t, ndim=1] {a_name}, int num_threads=0"
        checks = _emit_checks_1d(a_name)
        code = f"""
{_HEADER}

def {func_name}({sig}):
    cdef Py_ssize_t i, n
{checks}
    n = {a_name}.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef const double[:] {a_name}_mv = {a_name}
    cdef double[:] out_mv = out
    {_thread_default_stmt(default_threads)}
    if n < {THRESH} or num_threads == 1:
        for i in range(n):
            out_mv[i] = {expr_c}
    else:
        if num_threads > 0:
            omp_set_num_threads(num_threads)
        with nogil, parallel():
            for i in prange(n, schedule="{sched}"):
                out_mv[i] = {expr_c}
    return out
"""
        return code
    else:
        init_src = _ast.unparse(local_init)
        code = f"""
{_HEADER}

def {func_name}(int num_threads=0):
    cdef Py_ssize_t i, n
    cdef np.ndarray[np.double_t, ndim=1] {a_name} = np.asarray({init_src}, dtype=np.float64)
    n = {a_name}.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef const double[:] {a_name}_mv = {a_name}
    cdef double[:] out_mv = out
    {_thread_default_stmt(default_threads)}
    if n < {THRESH} or num_threads == 1:
        for i in range(n):
            out_mv[i] = {expr_c}
    else:
        if num_threads > 0:
            omp_set_num_threads(num_threads)
        with nogil, parallel():
            for i in prange(n, schedule="{sched}"):
                out_mv[i] = {expr_c}
    return out
"""
        return code

# ---- 一维单输出 ----
def _gen_single_output(excludes: set[str], func_name, for_node: _ast.For, assign: _ast.Assign, default_threads, schedule):
    idx_name = for_node.target.id if isinstance(for_node.target, _ast.Name) else "i"
    n_expr = _ast.unparse(for_node.iter.args[0]) if for_node.iter.args else "n"
    tgt = assign.targets[0]
    if not (isinstance(tgt, _ast.Subscript) and isinstance(tgt.value, _ast.Name)):
        raise ValueError("不支持的目标赋值形式（期望 out[i]）")
    out_name = tgt.value.id

    used_names = {n.id for n in _ast.walk(assign.value) if isinstance(n, _ast.Name)}
    input_arrays = sorted([x for x in used_names if x not in {idx_name, out_name}])

    params, param_names = _params_from_sizes(excludes, n_expr)
    allowed_names = {idx_name} | set(param_names)
    writer = _ExprWriter(allowed_names=allowed_names, allowed_arrays=set([out_name] + input_arrays))
    expr_c = writer.visit(assign.value)

    checks = "".join([_emit_checks_1d(x) for x in input_arrays])
    in_mv_decl = "\n    cdef const double[:] " + ", ".join([f"{x}_mv" for x in input_arrays]) + " = " + ", ".join(input_arrays) if input_arrays else ""

    sched = _schedule_literal(schedule)
    sig = f"{params}, {', '.join([f'np.ndarray[np.double_t, ndim=1] {x}' for x in input_arrays])}, int num_threads=0" if input_arrays else f"{params}, int num_threads=0"
    code = f"""
{_HEADER}

def {func_name}({sig}):
    cdef Py_ssize_t {idx_name}
{checks}
    cdef np.ndarray[np.double_t, ndim=1] {out_name} = np.empty({n_expr}, dtype=np.float64)
    {in_mv_decl}
    cdef double[:] {out_name}_mv = {out_name}
    {_thread_default_stmt(default_threads)}
    with nogil, parallel():
        for {idx_name} in prange({n_expr}, schedule="{sched}"):
            {out_name}_mv[{idx_name}] = {expr_c}
    return {out_name}
"""
    return code

# ---- 一维多输出 ----
def _gen_multi_output(excludes: set[str], func_name, for_node: _ast.For, assigns: list[_ast.Assign], default_threads, schedule):
    idx_name = for_node.target.id if isinstance(for_node.target, _ast.Name) else "i"
    n_expr = _ast.unparse(for_node.iter.args[0]) if for_node.iter.args else "n"

    out_names = []
    for a in assigns:
        if isinstance(a.targets[0], _ast.Subscript) and isinstance(a.targets[0].value, _ast.Name):
            out_names.append(a.targets[0].value.id)
        else:
            raise ValueError("不支持的目标赋值形式（期望 out[i]）")

    used = set()
    for a in assigns:
        used |= {n.id for n in _ast.walk(a.value) if isinstance(n, _ast.Name)}
    input_arrays = sorted([x for x in used if x not in set(out_names) | {idx_name}])

    params, param_names = _params_from_sizes(excludes, n_expr)
    allowed_names = {idx_name} | set(param_names)

    writer = _ExprWriter(allowed_names=allowed_names, allowed_arrays=set(out_names + input_arrays))
    assign_lines = []
    for a in assigns:
        expr = writer.visit(a.value)
        on = a.targets[0].value.id
        assign_lines.append(f"{on}_mv[{idx_name}] = {expr}")
    assign_body = "\n            ".join(assign_lines)

    checks = "".join([_emit_checks_1d(x) for x in input_arrays])
    decl_out_arrays = "\n    ".join([f"cdef np.ndarray[np.double_t, ndim=1] {o} = np.empty({n_expr}, dtype=np.float64)" for o in out_names])
    decl_in_mv = "\n    cdef const double[:] " + ", ".join([f"{x}_mv" for x in input_arrays]) + " = " + ", ".join(input_arrays) if input_arrays else ""
    decl_out_mvs = "\n    ".join([f"cdef double[:] {o}_mv = {o}" for o in out_names])

    sched = _schedule_literal(schedule)
    sig = f"{params}, {', '.join([f'np.ndarray[np.double_t, ndim=1] {x}' for x in input_arrays])}, int num_threads=0" if input_arrays else f"{params}, int num_threads=0"
    code = f"""
{_HEADER}

def {func_name}({sig}):
    cdef Py_ssize_t {idx_name}
{checks}
    {decl_out_arrays}
    {decl_in_mv}
    {decl_out_mvs}
    {_thread_default_stmt(default_threads)}
    with nogil, parallel():
        for {idx_name} in prange({n_expr}, schedule="{sched}"):
            {assign_body}
    return {', '.join(out_names)}
"""
    return code

# ---- 单层归约 ----
def _gen_reduction(excludes: set[str], func_name, n_expr: str, body_expr: _ast.AST, idx_name: str, reduce_var: str, for_scope: _ast.AST, default_threads, schedule, apply_sqrt_final: bool):
    params, param_names = _params_from_sizes(excludes, n_expr)
    allowed_names = {idx_name, reduce_var} | set(param_names)
    rhs_node = _inline_simple_names(body_expr, for_scope, allowed_names)
    writer = _ExprWriter(allowed_names=allowed_names, allowed_arrays=set())
    rhs = writer.visit(rhs_node)
    sched = _schedule_literal(schedule)
    ret_line = f"return sqrt({reduce_var})" if apply_sqrt_final else f"return {reduce_var}"
    code = f"""
{_HEADER}

def {func_name}({params}, int num_threads=0):
    cdef Py_ssize_t {idx_name}, t, tid
    cdef double {reduce_var} = 0.0
    cdef int PAD = 16
    {_thread_default_stmt(default_threads)}
    cdef int T = omp_get_max_threads()
    cdef np.ndarray[np.double_t, ndim=1] _locals = np.zeros(T * PAD, dtype=np.float64)
    cdef double[:] locals_mv = _locals
    with nogil, parallel():
        tid = omp_get_thread_num()
        for {idx_name} in prange({n_expr}, schedule="{sched}"):
            locals_mv[tid * PAD] += {rhs}
    for t in range(T):
        {reduce_var} += locals_mv[t * PAD]
    {ret_line}
"""
    return code

# ---- 多层 for 归约（含条件） ----
def _gen_reduction_nested(excludes: set[str], func_name, outer_for: _ast.For, aug_node: _ast.AugAssign, reduce_var: str, default_threads, schedule, apply_sqrt_final: bool):
    idx_names, ranges, _ = _collect_nested_for_indices(outer_for)
    params, param_names = _params_from_sizes(excludes, *ranges)
    allowed_names = set(idx_names) | {reduce_var} | set(param_names)

    rhs_node = _inline_simple_names(aug_node.value, outer_for, allowed_names)
    writer = _ExprWriter(allowed_names=allowed_names, allowed_arrays=set())
    rhs = writer.visit(rhs_node)

    cond_line = None
    if_node = _find_enclosing_if_with_node(outer_for, aug_node)
    if if_node is not None:
        test_node = _inline_simple_names(if_node.test, outer_for, allowed_names)
        test_expr = writer.visit(test_node)
        cond_line = f"if {test_expr}:"

    base = " " * 12
    inner_lines = []
    for depth, (name, rng) in enumerate(zip(idx_names[1:], ranges[1:])):
        inner_lines.append(f"{base}{' ' * (4*depth)}for {name} in range({rng}):")
    tail_indent = base + (' ' * (4*len(idx_names[1:])))
    if cond_line:
        inner_lines.append(f"{tail_indent}{cond_line}")
        inner_lines.append(f"{tail_indent}    locals_mv[tid * PAD] += {rhs}")
    else:
        inner_lines.append(f"{tail_indent}locals_mv[tid * PAD] += {rhs}")
    body_block = "\n".join(inner_lines) if inner_lines else f"{base}locals_mv[tid * PAD] += {rhs}"

    sched = _schedule_literal(schedule)
    n0 = ranges[0]
    idx_decl = ", ".join(idx_names) if idx_names else "i"
    ret_line = f"return sqrt({reduce_var})" if apply_sqrt_final else f"return {reduce_var}"

    code = f"""
{_HEADER}

def {func_name}({params}, int num_threads=0):
    cdef Py_ssize_t {idx_decl}, t, tid
    cdef double {reduce_var} = 0.0
    cdef int PAD = 16
    {_thread_default_stmt(default_threads)}
    cdef int T = omp_get_max_threads()
    cdef np.ndarray[np.double_t, ndim=1] _locals = np.zeros(T * PAD, dtype=np.float64)
    cdef double[:] locals_mv = _locals
    with nogil, parallel():
        tid = omp_get_thread_num()
        for {idx_names[0]} in prange({n0}, schedule="{sched}"):
{body_block}
    for t in range(T):
        {reduce_var} += locals_mv[t * PAD]
    {ret_line}
"""
    return code

# ---- 二维填充 ----
def _gen_square_fill_nested(excludes: set[str], func_name, arr_name, i_name, j_name, n0, n1, expr_node, default_threads, schedule):
    params, param_names = _params_from_sizes(excludes, n0, n1)
    allowed_names = {i_name, j_name} | set(param_names)
    writer = _ExprWriter(allowed_names=allowed_names, allowed_arrays={arr_name})
    expr = writer.visit(expr_node)
    sched = _schedule_literal(schedule)
    code = f"""
{_HEADER}

def {func_name}({params}, int num_threads=0):
    cdef Py_ssize_t {i_name}, {j_name}
    cdef np.ndarray[np.double_t, ndim=2] {arr_name} = np.empty(({n0}, {n1}), dtype=np.float64)
    cdef double[:, ::1] {arr_name}_mv = {arr_name}
    if num_threads > 0:
        omp_set_num_threads(num_threads)
    with nogil, parallel():
        for {i_name} in prange({n0}, schedule="{sched}"):
            for {j_name} in range({n1}):
                {arr_name}_mv[{i_name}, {j_name}] = {expr}
    return {arr_name}
"""
    return code

# ---- 二维逐元素 ----
# def _gen_2d_elementwise(excludes: set[str], func_name: str,
#                         out_name: str, i_name: str, j_name: str,
#                         n0: str, n1: str, expr_node: _ast.AST,
#                         inputs: set[str], default_threads, schedule):
#     allowed_arrays = set(inputs) | {out_name}
#     params, param_names = _params_from_sizes(excludes, n0, n1)
#     allowed_names = {i_name, j_name} | set(param_names)

#     writer = _ExprWriter(allowed_names=allowed_names, allowed_arrays=allowed_arrays)
#     expr = writer.visit(expr_node)
#     sched = _schedule_literal(schedule)

#     inputs_sorted = sorted(inputs)
#     checks = "".join([_emit_checks_2d(x) for x in inputs_sorted])
#     in_mvs = "\n    ".join([f"cdef const double[:, ::1] {x}_mv = {x}" for x in inputs_sorted]) if inputs_sorted else ""
#     inputs_sig = (", " + ", ".join([f"np.ndarray[np.double_t, ndim=2] {x}" for x in inputs_sorted])) if inputs_sorted else ""

#     code = f"""
# {_HEADER}

# def {func_name}({params}{inputs_sig}, int num_threads=0):
#     cdef Py_ssize_t {i_name}, {j_name}
# {checks}
#     cdef np.ndarray[np.double_t, ndim=2] {out_name} = np.empty(({n0}, {n1}), dtype=np.float64)
#     cdef double[:, ::1] {out_name}_mv = {out_name}
#     {in_mvs}
#     if num_threads > 0:
#         omp_set_num_threads(num_threads)
#     with nogil, parallel():
#         for {i_name} in prange({n0}, schedule="{sched}"):
#             for {j_name} in range({n1}):
#                 {out_name}_mv[{i_name}, {j_name}] = {expr}
#     return {out_name}
# """
#     return code
def _gen_2d_elementwise(excludes: set[str], func_name: str,
                        out_name: str, i_name: str, j_name: str,
                        n0: str, n1: str, expr_node: _ast.AST,
                        inputs: set[str], default_threads, schedule):
    # 参与 RHS 的数组 + 输出数组 都要允许下标访问
    allowed_arrays = set(inputs) | {out_name}

    # 不再把 M、N 暴露为形参；签名只包含所有输入数组 + num_threads
    inputs_sorted = sorted(inputs)
    inputs_sig = ", ".join([f"np.ndarray[np.double_t, ndim=2] {x}" for x in inputs_sorted])
    if inputs_sig:
        inputs_sig = inputs_sig + ", "

    # 允许的名字里保留循环索引 i/j（尺寸符号不再需要）
    allowed_names = {i_name, j_name}

    writer = _ExprWriter(allowed_names=allowed_names, allowed_arrays=allowed_arrays)
    expr = writer.visit(expr_node)
    sched = _schedule_literal(schedule)

    # dtype/布局检查 + memoryview
    checks = "".join([_emit_checks_2d(x) for x in inputs_sorted])
    in_mvs = "\n    ".join([f"cdef const double[:, ::1] {x}_mv = {x}" for x in inputs_sorted]) if inputs_sorted else ""

    # 选择第一个输入数组作为“尺寸来源”
    size_src = inputs_sorted[0] if inputs_sorted else None
    if size_src is None:
        # 极少见：表达式完全不使用输入数组（例如常量表达式），兜底用 out 的形状表达式 n0/n1
        # 这里直接报错更安全，也可按需要改成接受 M/N 形参
        raise ValueError("二维逐元素需要至少一个输入数组用于推导形状")

    code = f"""
{_HEADER}

def {func_name}({inputs_sig}int num_threads=0):
    cdef Py_ssize_t {i_name}, {j_name}
{checks}
    # 从第一个输入数组推导 M、N
    cdef Py_ssize_t M = {size_src}.shape[0]
    cdef Py_ssize_t N = {size_src}.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] {out_name} = np.empty((M, N), dtype=np.float64)
    cdef double[:, ::1] {out_name}_mv = {out_name}
    {in_mvs}
    if num_threads > 0:
        omp_set_num_threads(num_threads)
    with nogil, parallel():
        for {i_name} in prange(M, schedule="{sched}"):
            for {j_name} in range(N):
                {out_name}_mv[{i_name}, {j_name}] = {expr}
    return {out_name}
"""
    return code

# ------------------------------ 主入口 ------------------------------

def convert_python_to_cython_omp_v2(py_func_code: str, func_name: str = "compute", default_threads=None, schedule: str = "static") -> str:
    src = _dedent(py_func_code).strip()
    an = _Analyzer(src)
    excludes = set(_DEF_EXCLUDES_STATIC) | set(an.import_aliases)

    # 通用 GEMM（任意层序 + += / =C+C / 临时acc）
    mmg = _detect_matmul_generic(an.root)
    if mmg:
        return _gen_matmul_gemm_mkn(
            func_name, mmg["A"], mmg["B"], mmg["C"],
            "M", "K", "N",
            default_threads, schedule
        )

    # 2D 填充
    two_d = _detect_2d_fill_nested(an.root)
    if two_d:
        return _gen_square_fill_nested(excludes, func_name, two_d["arr"], two_d["i"], two_d["j"], two_d["n0"], two_d["n1"], two_d["expr"], default_threads, schedule)

    # ✅ 通用二维逐元素
    el2d = _detect_2d_elementwise(an.root)
    if el2d:
        return _gen_2d_elementwise(
            excludes, func_name,
            el2d["out"], el2d["i"], el2d["j"],
            el2d["n0"], el2d["n1"], el2d["expr"], el2d["inputs"],
            default_threads, schedule
        )

    # 列表推导
    lc = an.list_comp()
    if lc and lc["kind"] == "range":
        gen = lc["gen"]
        fake_for = _ast.For(target=gen.target, iter=gen.iter, body=[], orelse=[])
        target = _ast.Subscript(value=_ast.Name(id='out', ctx=_ast.Load()), slice=gen.target, ctx=_ast.Store())
        assign = _ast.Assign(targets=[target], value=lc["comp"].elt)
        return _gen_single_output(excludes, func_name, fake_for, assign, default_threads, schedule)
    if lc and lc["kind"] == "iterable":
        gen = lc["gen"]
        a_name = gen.iter.id
        local_init = an.local_inits.get(a_name)
        return _gen_list_comp_iterable(func_name, a_name, gen.target.id, lc["comp"].elt, local_init, default_threads, schedule)

    # 一维写数组
    mo = an.simple_for_multi_assign()
    if mo:
        return _gen_multi_output(excludes, func_name, mo["for"], mo["assigns"], default_threads, schedule)
    so = an.simple_for_assign()
    if so:
        return _gen_single_output(excludes, func_name, so["for"], so["assign"], default_threads, schedule)

    # 归约（多层优先）
    red_var = an.reduction_var()
    if red_var:
        apply_sqrt = _need_final_sqrt(an.fun or an.root, red_var)
        for node in _ast.walk(an.root):
            if isinstance(node, _ast.For) and isinstance(node.iter, _ast.Call) and isinstance(node.iter.func, _ast.Name) and node.iter.func.id == 'range':
                aug = None
                for s in _ast.walk(node):
                    if isinstance(s, _ast.AugAssign) and isinstance(s.op, _ast.Add) and isinstance(s.target, _ast.Name) and s.target.id == red_var:
                        aug = s
                        break
                if aug is not None:
                    return _gen_reduction_nested(excludes, func_name, node, aug, red_var, default_threads, schedule, apply_sqrt)
        for n in _ast.walk(an.root):
            if isinstance(n, _ast.For) and isinstance(n.iter, _ast.Call) and isinstance(n.iter.func, _ast.Name) and n.iter.func.id == 'range':
                rhs = None
                for s in _ast.walk(n):
                    if isinstance(s, _ast.AugAssign) and isinstance(s.op, _ast.Add) and isinstance(s.target, _ast.Name) and s.target.id == red_var:
                        rhs = s.value
                if rhs is not None:
                    n_expr = _ast.unparse(n.iter.args[0]) if n.iter.args else "n"
                    idx_name = n.target.id if isinstance(n.target, _ast.Name) else "i"
                    return _gen_reduction(excludes, func_name, n_expr, rhs, idx_name, red_var, n, default_threads, schedule, apply_sqrt)

    # 兜底
    return f"""
{_HEADER}

def {func_name}(int n, int num_threads=0):
    if num_threads > 0:
        omp_set_num_threads(num_threads)
    # 未识别的模式：请调整代码或扩展识别器
    return None
"""
