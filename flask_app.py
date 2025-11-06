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
    f_str = "unknown"
    method = "symbolic"
    rows = []

    try:
        # === METODE SIMBOLIK ===
        expr_simp = sp.simplify(expr)
        expr_exp = sp.expand(expr_simp)
        f_expr = -sp.diff(expr_exp, t)        # f(t) = -dR/dt
        h_expr = f_expr / expr_exp            # h(t) = f(t)/R(t)
        
        h_expr = sp.simplify(h_expr)
        f_expr = sp.simplify(f_expr)

        for t_val in t_values:
            try:
                h_val = float(h_expr.subs(t, t_val).evalf())
                f_val = float(f_expr.subs(t, t_val).evalf())

                if not np.isfinite(h_val) or h_val < 0:
                    h_val = 0.0
                if not np.isfinite(f_val) or f_val < 0:
                    f_val = 0.0
            except:
                h_val = 0.0
                f_val = 0.0

            rows.append({
                "t": f"{t_val:.6e}",
                "hazard_rate": f"{h_val:.6e}",
                "failure_density": f"{f_val:.6e}",
                **lambdas
            })

        h_str = str(h_expr)
        f_str = str(f_expr)
        method = "symbolic (evalf)"

    except Exception as e_sym:
        # === METODE NUMERIK (jika simbolik gagal) ===
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
                    f_val = 0.0
                elif R_t <= 0.0:
                    h_val = np.inf
                    f_val = 0.0
                else:
                    delta = max(t_val * delta_ratio, 1e-8)
                    t_plus = t_val + delta
                    t_minus = max(t_val - delta, 1e-8)
                    R_plus = R_func(t_plus)
                    R_minus = R_func(t_minus)

                    # Clip untuk stabilitas
                    R_plus = np.clip(R_plus, 1e-12, 0.999999999999)
                    R_minus = np.clip(R_minus, 1e-12, 0.999999999999)
                    R_t_clipped = np.clip(R_t, 1e-12, 0.999999999999)

                    dR_dt = (R_plus - R_minus) / (2 * delta)
                    f_val = max(-dR_dt, 0)                     # f(t) = -dR/dt
                    h_val = max(-dR_dt / R_t_clipped, 0)       # h(t) = -dR/dt / R(t)

                h_val = 0.0 if not np.isfinite(h_val) else h_val
                f_val = 0.0 if not np.isfinite(f_val) else f_val

            except Exception as e_num:
                print(f"Numerical error at t={t_val}: {e_num}")
                h_val = 0.0
                f_val = 0.0

            rows.append({
                "t": f"{t_val:.6e}",
                "hazard_rate": f"{h_val:.6e}" if np.isfinite(h_val) else "0.000000e+00",
                "failure_density": f"{f_val:.6e}" if np.isfinite(f_val) else "0.000000e+00",
                **lambdas
            })

        h_str = "numerical: h(t) ≈ -dR/dt / R(t)"
        f_str = "numerical: f(t) ≈ -dR/dt"

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