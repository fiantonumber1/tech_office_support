from flask import Flask, request, jsonify
import sympy as sp
import pandas as pd
from mpmath import mp, mpf

# Presisi cukup untuk 0.0000395 * 31200 = 1.23
mp.dps = 20

app = Flask(__name__)

def safe_float(val, threshold=1e-40):
    """Konversi mpf ke float, tapi jangan jadi 0 karena underflow"""
    if val == 0 or abs(val) < mpf(threshold):
        return 0.0
    try:
        f = float(val)
        if abs(f) < 1e-40:
            return 0.0
        return f
    except:
        return 0.0

def format_scientific(val):
    """Format float ke .6e"""
    if val == 0:
        return "0.000000e+00"
    return f"{val:.6e}"

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
        # === METODE SIMBOLIK ===
        expr_simp = sp.simplify(expr)
        expr_exp = sp.expand(expr_simp)
        f_expr = -sp.diff(expr_exp, t)
        h_expr = f_expr / expr_exp
        
        h_expr = sp.simplify(h_expr)
        f_expr = sp.simplify(f_expr)

        h_func = sp.lambdify(t, h_expr, modules='mpmath')
        f_func = sp.lambdify(t, f_expr, modules='mpmath')

        for t_val in t_values:
            t_mp = mpf(t_val)
            try:
                h_val = safe_float(h_func(t_mp))
                f_val = safe_float(f_func(t_mp))
            except Exception as e:
                print(f"Error: {e}")
                h_val = f_val = 0.0

            rows.append({
                "t": f"{t_val:.6e}",
                "hazard_rate": format_scientific(h_val),
                "failure_density": format_scientific(f_val),
                **lambdas
            })

        h_str = str(h_expr)
        f_str = str(f_expr)
        method = "symbolic (mpmath)"

    except Exception as e_sym:
        # === METODE NUMERIK ===
        method = "numerical"
        print(f"Symbolic failed: {e_sym}")

        expr_num = expr.subs(lambdas)
        R_func = sp.lambdify(t, expr_num, modules='mpmath')

        for t_val in t_values:
            t_mp = mpf(t_val)
            try:
                R_t = R_func(t_mp)
                R_t = safe_float(R_t)
                if R_t >= 1.0:
                    h_val = f_val = 0.0
                elif R_t <= 0.0:
                    h_val = float('inf')
                    f_val = 0.0
                else:
                    delta = max(R_t * 1e-8, 1e-20)
                    t_plus = t_val + delta
                    t_minus = max(t_val - delta, 1e-20)

                    R_plus = safe_float(R_func(mpf(t_plus)))
                    R_minus = safe_float(R_func(mpf(t_minus)))

                    dR_dt = (R_plus - R_minus) / (2 * delta)
                    f_val = max(-dR_dt, 0)
                    h_val = f_val / R_t if R_t > 0 else 0.0

            except Exception as e_num:
                print(f"Numerical error: {e_num}")
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


@app.route('/calculate_hazard', methods=['POST'])
def calc():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON"}), 400

    fungsi = data.get('fungsi', '1')
    lambdas = data.get('lambdas', {})
    t_values = data.get('t_values', [1000])

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5632, debug=True)