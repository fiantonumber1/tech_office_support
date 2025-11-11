from flask import Flask, request, jsonify
import sympy as sp
import pandas as pd
import re

app = Flask(__name__)

def format_scientific(val):
    if val == 0 or abs(val) < 1e-40:
        return "0.000000e+00"
    return f"{val:.6e}"

# === SAFE SIMPLIFY WITH TIMEOUT ===
def safe_simplify(expr, timeout=20):
    """
    Simplify expression with SymPy timeout (in seconds).
    Falls back to original expression if timeout or error.
    """
    try:
        return sp.simplify(expr, timeout=timeout)
    except TimeoutError:
        print(f"Simplify timed out after {timeout}s. Using unsimplified expression.")
        return expr
    except Exception as e:
        print(f"Simplify error: {e}. Using unsimplified.")
        return expr

# === SAFE EXPAND WITH TIMEOUT ===
def safe_expand(expr, timeout=15):
    try:
        return sp.expand(expr, timeout=timeout)
    except TimeoutError:
        print(f"Expand timed out after {timeout}s. Using original.")
        return expr
    except Exception as e:
        print(f"Expand error: {e}. Using original.")
        return expr


def calculate_reliability(fungsi_str, lambdas, t_values):
    t = sp.symbols('t')
    lam_symbols = {name: sp.symbols(name) for name in lambdas.keys()}
    locals_dict = {**lam_symbols, 'exp': sp.exp}
    
    try:
        expr = sp.sympify(fungsi_str, locals=locals_dict)
    except Exception as e:
        raise ValueError(f"Parse error: {e}")

    R_str = fungsi_str
    h_str = "unknown"
    f_str = "unknown"
    method = "numerical"
    rows = []

    # === METODE SIMBOLIK LANGSUNG ===
    def symbolic_direct():
        try:
            expr_simp = safe_simplify(expr, timeout=60)
            expr_exp = safe_expand(expr_simp, timeout=60)
            f_expr = -sp.diff(expr_exp, t)
            h_expr = f_expr / expr_exp

            h_expr = safe_simplify(h_expr, timeout=60)
            f_expr = safe_simplify(f_expr, timeout=60)

            h_expr_num = h_expr.subs(lambdas)
            f_expr_num = f_expr.subs(lambdas)

            h_func = sp.lambdify(t, h_expr_num, modules='numpy')
            f_func = sp.lambdify(t, f_expr_num, modules='numpy')

            local_rows = []
            for t_val in t_values:
                try:
                    h_val = float(h_func(t_val))
                    f_val = float(f_func(t_val))
                    if abs(h_val) < 1e-40: h_val = 0.0
                    if abs(f_val) < 1e-40: f_val = 0.0
                except Exception as e:
                    print(f"Symbolic eval error at t={t_val}: {e}")
                    h_val = f_val = 0.0
                local_rows.append({
                    "t": f"{t_val:.6e}",
                    "hazard_rate": format_scientific(h_val),
                    "failure_density": format_scientific(f_val),
                    **lambdas
                })
            return {
                "h_str": str(h_expr),
                "f_str": str(f_expr),
                "rows": local_rows,
                "method": "symbolic"
            }
        except Exception as e:
            raise RuntimeError(f"Symbolic direct failed: {e}")

    # === METODE LOG (R(t) > 0) ===
    def symbolic_log():
        try:
            logR = sp.log(expr)
            h_expr = -sp.diff(logR, t)
            f_expr = h_expr * expr

            h_expr = safe_simplify(h_expr, timeout=60)
            f_expr = safe_simplify(f_expr, timeout=60)

            h_expr_num = h_expr.subs(lambdas)
            f_expr_num = f_expr.subs(lambdas)

            h_func = sp.lambdify(t, h_expr_num, 'numpy')
            f_func = sp.lambdify(t, f_expr_num, 'numpy')

            local_rows = []
            for t_val in t_values:
                try:
                    h_val = float(h_func(t_val))
                    f_val = float(f_func(t_val))
                    if abs(h_val) < 1e-40: h_val = 0.0
                    if abs(f_val) < 1e-40: f_val = 0.0
                except:
                    h_val = f_val = 0.0
                local_rows.append({
                    "t": f"{t_val:.6e}",
                    "hazard_rate": format_scientific(h_val),
                    "failure_density": format_scientific(f_val),
                    **lambdas
                })
            return {
                "h_str": str(h_expr),
                "f_str": str(f_expr),
                "rows": local_rows,
                "method": "symbolic_log"
            }
        except Exception as e:
            raise RuntimeError(f"Log method failed: {e}")

    # === COBA METODE SIMBOLIK LANGSUNG ===
    try:
        print("Trying symbolic direct method...")
        result = symbolic_direct()
        h_str = result["h_str"]
        f_str = result["f_str"]
        rows = result["rows"]
        method = result["method"]
    except Exception as e:
        print(f"Symbolic direct failed: {e}. Trying log method...")

    # === JIKA GAGAL → COBA LOG METHOD ===
    if method not in ("symbolic", "symbolic_log"):
        try:
            print("Trying symbolic log method...")
            result = symbolic_log()
            h_str = result["h_str"]
            f_str = result["f_str"]
            rows = result["rows"]
            method = result["method"]
        except Exception as e_log:
            print(f"Log method failed: {e_log}. Falling back to numerical.")

    # === JIKA MASIH GAGAL → NUMERICAL ===
    if method not in ("symbolic", "symbolic_log"):
        method = "numerical"
        print("Using numerical differentiation...")
        expr_num = expr.subs(lambdas)
        R_func = sp.lambdify(t, expr_num, modules='numpy')

        for t_val in t_values:
            try:
                R_t = float(R_func(t_val))
                if R_t >= 1.0:
                    h_val = f_val = 0.0
                elif R_t <= 0.0:
                    h_val = float('inf')
                    f_val = 0.0
                else:
                    delta = max(t_val * 1e-8, 1e-12)
                    t_plus = t_val + delta
                    t_minus = max(t_val - delta, 1e-12)

                    R_plus = float(R_func(t_plus))
                    R_minus = float(R_func(t_minus))

                    dR_dt = (R_plus - R_minus) / (2 * delta)
                    f_val = max(-dR_dt, 0)
                    h_val = f_val / R_t if R_t > 0 else 0.0

                if abs(h_val) < 1e-40: h_val = 0.0
                if abs(f_val) < 1e-40: f_val = 0.0

            except Exception as e_num:
                print(f"Numerical error at t={t_val}: {e_num}")
                h_val = f_val = 0.0

            rows.append({
                "t": f"{t_val:.6e}",
                "hazard_rate": format_scientific(h_val),
                "failure_density": format_scientific(f_val),
                **lambdas
            })

        h_str = "numerical"
        f_str = "numerical"

    return {
        "R": R_str,
        "h": h_str,
        "f": f_str,
        "data": pd.DataFrame(rows).to_dict(orient="records"),
        "method": method
    }


def invert_by_low_order_taylor(r_target,R_t_str, order=2, do_subs=None):
    """
    Membalik R(t) menjadi t(R) dengan pendekatan deret Taylor orde rendah.

    Parameters
    ----------
    R_t_str : str
        Fungsi reliabilitas, mis. "exp(-lam*t)" atau fungsi kompleks.
    order : int
        Orde deret Taylor (1 = linear, 2 = kuadrat, dst.)
    do_subs : dict (optional)
        Substitusi nilai numerik, mis. {'lam': 3e-5}.

    Returns
    -------
    dict dengan:
        - 't_chosen' : solusi simbolik utama (yang realistis)
        - 't_chosen_subs' : solusi numerik (jika do_subs diberikan)
        - 'R_series' : ekspansi Taylor R(t)
        - 'R_expr' : ekspresi asli R(t)
    """
    t, R = sp.symbols('t R')

    # Deteksi simbol lambda dan variabel lain
    local_names = {'t': t, 'R': R, 'exp': sp.exp}
    found = set(re.findall(r'lam[A-Za-z0-9_]*', R_t_str))
    if 'lam' in R_t_str and 'lam' not in found:
        found.add('lam')
    found.update(set(re.findall(r'[A-Z]', R_t_str)))
    for name in found:
        local_names[name] = sp.symbols(name)

    # Parse ekspresi
    expr_str = R_t_str.replace('^', '**').replace('\n', '').strip()
    R_expr = sp.sympify(expr_str, locals=local_names)

    # Deret Taylor sekitar t=0
    R_series = sp.series(R_expr, t, 0, order + 1).removeO().expand()

    # Pecahkan R_series == R untuk t
    sols = sp.solve(sp.Eq(R_series, R), t)

    # Pilih solusi yang paling "masuk akal": real, positif, kecil untuk R≈1
    best_sol = None
    min_abs_val = float('inf')
    test_R = 0.99
    for s in sols:
        try:
            s_val = s.subs({R: test_R})
            if s_val.is_real and s_val > 0:
                abs_val = abs(s_val)
                if abs_val < min_abs_val:
                    min_abs_val = abs_val
                    best_sol = s
        except Exception:
            pass
    if best_sol is None and sols:
        best_sol = sols[0]

    # Substitusi numerik jika diminta
    t_chosen_subs = None
    if do_subs and best_sol is not None:
        try:
            t_chosen_subs = sp.simplify(best_sol.subs(do_subs))
        except Exception:
            t_chosen_subs = None

    return {
        't_chosen': best_sol,
        't_chosen_subs': t_chosen_subs
    }

def invert_by_low_order_taylor(R_t_str, order=2, do_subs=None):
    """
    Membalik R(t) menjadi t(R) dengan pendekatan deret Taylor orde rendah.

    Parameters
    ----------
    R_t_str : str
        Fungsi reliabilitas, mis. "exp(-lam*t)" atau fungsi kompleks.
    order : int
        Orde deret Taylor (1 = linear, 2 = kuadrat, dst.)
    do_subs : dict (optional)
        Substitusi nilai numerik, mis. {'lam': 3e-5}.

    Returns
    -------
    dict dengan:
        - 't_chosen' : solusi simbolik utama (yang realistis)
        - 't_chosen_subs' : solusi numerik (jika do_subs diberikan)
        - 'R_series' : ekspansi Taylor R(t)
        - 'R_expr' : ekspresi asli R(t)
    """
    t, R = sp.symbols('t R')

    # Deteksi simbol lambda dan variabel lain
    local_names = {'t': t, 'R': R, 'exp': sp.exp}
    found = set(re.findall(r'lam[A-Za-z0-9_]*', R_t_str))
    if 'lam' in R_t_str and 'lam' not in found:
        found.add('lam')
    found.update(set(re.findall(r'[A-Z]', R_t_str)))
    for name in found:
        local_names[name] = sp.symbols(name)

    # Parse ekspresi
    expr_str = R_t_str.replace('^', '**').replace('\n', '').strip()
    R_expr = sp.sympify(expr_str, locals=local_names)

    # Deret Taylor sekitar t=0
    R_series = sp.series(R_expr, t, 0, order + 1).removeO().expand()

    # Pecahkan R_series == R untuk t
    sols = sp.solve(sp.Eq(R_series, R), t)

    # Pilih solusi yang paling "masuk akal": real, positif, kecil untuk R≈1
    best_sol = None
    min_abs_val = float('inf')
    test_R = 0.99
    for s in sols:
        try:
            s_val = s.subs({R: test_R})
            if s_val.is_real and s_val > 0:
                abs_val = abs(s_val)
                if abs_val < min_abs_val:
                    min_abs_val = abs_val
                    best_sol = s
        except Exception:
            pass
    if best_sol is None and sols:
        best_sol = sols[0]

    # Substitusi numerik jika diminta
    t_chosen_subs = None
    if do_subs and best_sol is not None:
        try:
            t_chosen_subs = sp.simplify(best_sol.subs(do_subs))
        except Exception:
            t_chosen_subs = None

    return {
        't_chosen': best_sol,
        't_chosen_subs': t_chosen_subs
    }

@app.route('/calculate_hazard', methods=['POST'])
def calc():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON"}), 400

    fungsi = data.get('fungsi', '1')
    lambdas = data.get('lambdas', {})
    t_values = data.get('t_values', [1000])
    r_target_to_find_t = data.get('r_target', 0.9)  # Default 0.9

    try:
        t_values = [float(t) for t in t_values]
        if any(t <= 0 for t in t_values):
            return jsonify({"error": "t > 0"}), 400
    except:
        return jsonify({"error": "Invalid t_values"}), 400

    try:
        result = calculate_reliability(fungsi, lambdas, t_values)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/calculate_t_r', methods=['POST'])
def calc_t_r():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON"}), 400

    fungsi = data.get('fungsi', '1')
    lambdas = data.get('lambdas', {})
    t_values = data.get('t_values', [1000])
    ordo = data.get('ordo', 2)
    r_target_to_find_t = data.get('r_target', 0.9)  # Default 0.9

    try:
        t_values = [float(t) for t in t_values]
        if any(t <= 0 for t in t_values):
            return jsonify({"error": "t > 0"}), 400
    except:
        return jsonify({"error": "Invalid t_values"}), 400

    try:
        result = {}
        # --- INVERSE CALCULATION: Find t such that R(t) = r_target ---
        if 0 < r_target_to_find_t < 1:
            subs_dict = lambdas.copy()
            subs_dict["R"] = r_target_to_find_t
            try:
                inv = invert_by_low_order_taylor(fungsi, order=ordo, do_subs=subs_dict)
                result['t_expression'] = str(inv['t_chosen'])
                result['t_value'] = float(inv['t_chosen_subs'])
            except Exception as e_inv:
                result['t_for_R'] = {
                    'R_target': r_target_to_find_t,
                    'error': f"Inversion failed: {str(e_inv)}"
                }
        else:
            result['t_for_R'] = {"error": "r_target must be between 0 and 1"}

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5632, debug=True)