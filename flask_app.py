from flask import Flask, request, jsonify
import sympy as sp
import numpy as np
import pandas as pd

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
    method = "symbolic"

    try:
        # BANTU SYMPY
        expr_simp = sp.simplify(expr)
        expr_exp = sp.expand(expr_simp)
        h_expr = -sp.diff(expr_exp, t) / expr_exp

        rows = []
        for t_val in t_values:
            try:
                # PAKAI evalf() → PASTI SIMBOLIK
                h_val = float(h_expr.subs(t, t_val).evalf())
                if not np.isfinite(h_val) or h_val < 0:
                    raise ValueError()
            except:
                h_val = 0.0

            rows.append({
                "t": f"{t_val:.6e}",
                "hazard_rate": f"{h_val:.6e}",
                **lambdas
            })

        h_str = str(h_expr)
        method = "symbolic (evalf)"

    except Exception as e_sym:
        method = "numerical"
        print(f"Symbolic evalf failed: {e_sym}. Using numerical diff.")

        expr_num = expr.subs(lambdas)
        R_func_raw = sp.lambdify(t, expr_num, modules=["numpy", {"exp": safe_exp}])
        def R_func(t_val):
            return safe_R_eval(R_func_raw, t_val)

        delta_ratio = 1e-6
        rows = []
        for t_val in t_values:
            try:
                R_t = R_func(t_val)
                if R_t >= 1.0:
                    h_val = 0.0
                elif R_t <= 0.0:
                    h_val = np.inf
                else:
                    delta = max(t_val * delta_ratio, 1e-8)
                    t_plus = t_val + delta
                    t_minus = max(t_val - delta, 1e-8)
                    R_plus = R_func(t_plus)
                    R_minus = R_func(t_minus)

                    if not (0 < R_plus < 1 and 0 < R_minus < 1):
                        h_val = np.inf if R_t < 1e-8 else 0.0
                    else:
                        dR_dt = (R_plus - R_minus) / (2 * delta)
                        h_val = max(-dR_dt / R_t, 0)

                h_val = h_val if np.isfinite(h_val) else 0.0
            except:
                h_val = 0.0

            rows.append({
                "t": f"{t_val:.6e}",
                "hazard_rate": f"{h_val:.6e}" if np.isfinite(h_val) else "0.000000e+00",
                **lambdas
            })

        h_str = "numerical: h(t) ≈ -dR/dt / R(t)"

    return {
        "R": R_str,
        "h": h_str,
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
        result = calculate_reliability(fungsi, lambdas, t_values)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5632, debug=True)