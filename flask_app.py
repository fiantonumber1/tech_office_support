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



def invert_by_low_order_taylor(r_target, R_t_str, order=2, do_subs=None):
    """
    Inversi R(t) → t menggunakan Taylor orde rendah.
    Mendukung do_subs dengan string key → otomatis konversi ke Symbol.
    """
    t, R = sp.symbols('t R', positive=True)

    # === 1. DETEKSI SEMUA SIMBOL (lambda + huruf kapital) ===
    local_names = {'t': t, 'R': R, 'exp': sp.exp}
    found_lam = set(re.findall(r'lam[A-Za-z0-9_]*', R_t_str))
    if 'lam' in R_t_str and not found_lam:
        found_lam.add('lam')
    found_caps = set(re.findall(r'[A-Z][A-Za-z0-9_]*', R_t_str))
    found = found_lam.union(found_caps)

    # Simpan mapping string → Symbol
    symbol_map = {}
    for name in found:
        sym = sp.symbols(name, positive=True)
        local_names[name] = sym
        symbol_map[name] = sym

    # === 2. PARSE R(t) ===
    expr_str = R_t_str.replace('^', '**').replace('\n', '').strip()
    try:
        R_expr = sp.sympify(expr_str, locals=local_names)
    except Exception as e:
        raise ValueError(f"Parse error: {e}")

    # === 3. TAYLOR SERIES di t=0 ===
    try:
        R_series = sp.series(R_expr, t, 0, order + 1).removeO().expand()
    except Exception as e:
        raise ValueError(f"Taylor error: {e}")

    # === 4. SOLVE R_series = R untuk t ===
    eq = sp.Eq(R_series, R)
    try:
        sols = sp.solve(eq, t)
    except:
        sols = []

    # === 5. PILIH SOLUSI POSITIF TERKECIL ===
    best_sol = None
    min_pos_val = float('inf')
    for s in sols:
        try:
            s_num = s.subs(R, r_target)
            if s_num.is_real:
                val = float(s_num)
                if val > 1e-12 and val < min_pos_val:
                    min_pos_val = val
                    best_sol = s
        except:
            continue

    if best_sol is None and sols:
        best_sol = sols[0]

    # === 6. SUBSTITUSI NUMERIK AMAN ===
    t_value = None
    if do_subs is not None and best_sol is not None:
        try:
            # Bangun full_subs: string → Symbol otomatis
            full_subs = {}
            for k, v in do_subs.items():
                if isinstance(k, str) and k in symbol_map:
                    full_subs[symbol_map[k]] = v
                else:
                    full_subs[k] = v
            full_subs[R] = r_target  # R pasti Symbol

            # DEBUG (bisa dihapus di produksi)
            print(f"[DEBUG] full_subs: {full_subs}")
            print(f"[DEBUG] best_sol: {best_sol}")

            # Substitusi
            expr = best_sol.subs(full_subs)
            print(f"[DEBUG] setelah subs: {expr}")

            # Cek apakah masih ada simbol
            free_syms = expr.free_symbols
            if free_syms:
                print(f"[ERROR] Simbol belum diganti: {free_syms}")
                t_value = None
            else:
                ev = expr.evalf()
                print(f"[DEBUG] evalf(): {ev} (type: {type(ev)})")

                if ev.is_real and ev.is_finite:
                    t_value = float(ev)
                    print(f"[SUCCESS] t_value: {t_value}")
                else:
                    print(f"[ERROR] Hasil tidak real/finite: {ev}")
                    t_value = None

        except Exception as e:
            print(f"Subs error: {e}")
            import traceback
            traceback.print_exc()
            t_value = None

    return {
        't_expression': best_sol,
        't_value': t_value  # float atau None
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
                inv = invert_by_low_order_taylor(r_target_to_find_t,fungsi, order=ordo, do_subs=subs_dict)
                result['t_expression'] = str(inv['t_expression'])
                result['t_value'] = float(inv['t_value'])
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