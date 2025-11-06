from flask import Flask, request, jsonify
import sympy as sp
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from mpmath import mp

# Set presisi global
getcontext().prec = 80
mp.dps = 80

app = Flask(__name__)

def safe_exp(x):
    x = np.asarray(x)
    return np.exp(-np.clip(x, -700, 700))

def safe_R_eval(R_func, t_val):
    try:
        result = float(R_func(t_val))
        if not np.isfinite(result):
            return 1.0
        return max(min(result, 1.0), 0.0)
    except:
        return 1.0

def to_scientific_str(val, default="0.000000e+00"):
    """Konversi angka (mpf/Decimal/float) ke string .6e"""
    if val == 0:
        return default
    try:
        # Gunakan mpmath untuk format akurat
        return f"{float(val):.6e}"
    except:
        return default

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
    method = "symbolic"
    rows = []

    try:
        # === METODE SIMBOLIK (presisi tinggi dengan mpmath) ===
        expr_simp = sp.simplify(expr)
        expr_exp = sp.expand(expr_simp)
        f_expr = -sp.diff(expr_exp, t)
        h_expr = f_expr / expr_exp
        
        h_expr = sp.simplify(h_expr)
        f_expr = sp.simplify(f_expr)

        for t_val in t_values:
            try:
                # Gunakan mpmath untuk evaluasi presisi tinggi
                t_mp = mp.mpf(t_val)
                h_mp = sp.lambdify(t, h_expr, modules='mpmath')(t_mp)
                f_mp = sp.lambdify(t, f_expr, modules='mpmath')(t_mp)

                h_val = float(h_mp) if abs(h_mp) >= 1e-60 else 0.0
                f_val = float(f_mp) if abs(f_mp) >= 1e-60 else 0.0
            except:
                h_val = f_val = 0.0

            rows.append({
                "t": f"{t_val:.6e}",
                "hazard_rate": to_scientific_str(h_val),
                "failure_density": to_scientific_str(f_val),
                **lambdas
            })

        h_str = str(h_expr)
        f_str = str(f_expr)
        method = "symbolic (mpmath)"

    except Exception as e_sym:
        # === METODE NUMERIK (presisi tinggi dengan mpmath) ===
        method = "numerical"
        print(f"Symbolic failed: {e_sym}. Using mpmath numerical diff.")

        expr_num = expr.subs(lambdas)
        R_func_mp = sp.lambdify(t, expr_num, modules='mpmath')

        rows = []
        for t_val in t_values:
            try:
                t_mp = mp.mpf(t_val)
                R_t = float(R_func_mp(t_mp))
                if R_t >= 1.0:
                    h_val = f_val = 0.0
                elif R_t <= 0.0:
                    h_val = float('inf')
                    f_val = 0.0
                else:
                    # Delta adaptif dengan mpmath
                    eps = mp.mpf('1e-12')
                    delta = max(mp.mpf(R_t) * eps, mp.mpf('1e-70'))
                    t_plus = t_mp + delta
                    t_minus = mp.max(t_mp - delta, mp.mpf('1e-70'))

                    R_plus = R_func_mp(t_plus)
                    R_minus = R_func_mp(t_minus)

                    R_plus = mp.nstr(mp.mpf(R_plus), 15)
                    R_minus = mp.nstr(mp.mpf(R_minus), 15)
                    R_t_clipped = mp.mpf(max(R_t, 1e-80))

                    dR_dt = (mp.mpf(R_plus) - mp.mpf(R_minus)) / (2 * delta)
                    f_val = float(-dR_dt) if -dR_dt > 0 else 0.0
                    h_val = float(-dR_dt / R_t_clipped) if R_t_clipped > 0 else 0.0

                h_val = 0.0 if abs(h_val) < 1e-60 else h_val
                f_val = 0.0 if abs(f_val) < 1e-60 else f_val

            except Exception as e_num:
                print(f"Numerical error at t={t_val}: {e_num}")
                h_val = f_val = 0.0

            rows.append({
                "t": f"{t_val:.6e}",
                "hazard_rate": to_scientific_str(h_val),
                "failure_density": to_scientific_str(f_val),
                **lambdas
            })

        h_str = "numerical: h(t) ≈ -dR/dt / R(t) [mpmath]"
        f_str = "numerical: f(t) ≈ -dR/dt [mpmath]"

    return {
        "R": R_str,
        "h": h_str,
        "f": f_str,
        "data": pd.DataFrame(rows).to_dict(orient="records"),
        "method": method
    }


@app.route('/calculate_hazard', methods=['POST'])
def calc():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON"}), 400

    fungsi = data.get('fungsi', '1')
    lambdas = data.get('lambdas', {})
    t_values = data.get('t_values', [1000])

    # Validasi t_values
    try:
        t_values = [float(t) for t in t_values]
        if any(t <= 0 for t in t_values):
            return jsonify({"error": "All t_values must be > 0"}), 400
    except:
        return jsonify({"error": "Invalid t_values"}), 400

    try:
        result = calculate_reliability(fungsi, lambdas, t_values)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5632, debug=True)