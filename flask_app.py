# flask_app.py
from flask import Flask, request, jsonify
import sympy as sp
from sympy import exp, diff, simplify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

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
    lam_symbols = {f"lam{i+1}": sp.symbols(f"lam{i+1}") for i in range(len(lambdas))}

    expr = sp.sympify(fungsi_str)
    for name, val in lambdas.items():
        sym = lam_symbols.get(name)
        if sym:
            expr = expr.subs(sym, val)

    R_str = str(expr)
    if len(R_str) > 200:
        R_str = R_str[:200] + "..."
    h_str = "unknown"
    method = "symbolic"

    # --- COBA SIMBOLIK ---
    try:
        R = simplify(expr, seconds=2)
        h_expr = simplify(-diff(R, t) / R, seconds=2)
        h_func = sp.lambdify(t, h_expr, modules=["numpy", {"exp": safe_exp}])

        rows = []
        for t_val in t_values:
            try:
                h_val = float(h_func(t_val))
                if np.isfinite(h_val) and h_val >= 0:
                    rows.append({
                        "t": f"{t_val:.6e}",
                        "hazard_rate": f"{h_val:.6e}",
                        **lambdas
                    })
                    continue
            except:
                pass
            raise
        h_str = str(h_expr)[:200] + "..." if len(str(h_expr)) > 200 else str(h_expr)

    except Exception as e_sym:
        # --- FALLBACK: Numerical derivative dari R_simp yang sudah disederhanakan ---
        method = "numerical (simplified symbolic + finite diff)"
        print(f"Symbolic hazard failed: {e_sym}. Using simplified R(t) + numerical diff.")

        R_func_raw = sp.lambdify(t, expr, modules=["numpy", {"exp": safe_exp}])
        def R_func(t_val):
            return safe_R_eval(R_func_raw, t_val)

        delta_ratio = 1e-6
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

                    if R_plus >= 1.0 or R_minus >= 1.0:
                        h_val = 0.0
                    elif R_plus <= 0 or R_minus <= 0:
                        h_val = np.inf
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

    df = pd.DataFrame(rows)
    return {
        "R": R_str,
        "h": h_str,
        "data": df.to_dict(orient="records"),
        "method": method
    }

@app.route('/calculate_hazard', methods=['POST'])
def calc():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    fungsi = data.get('fungsi', '1')
    lambdas = data.get('lambdas', {})
    t_values = data.get('t_values', [1000, 10000, 50000])
    title = data.get('title', 'Hazard Rate')

    try:
        result = calculate_reliability(fungsi, lambdas, t_values)

        if data.get('include_plot', False):
            try:
                plt.figure(figsize=(10, 6))
                df = pd.DataFrame(result['data'])
                t_vals = df['t'].astype(float)
                h_vals = df['hazard_rate'].astype(float)
                mask = np.isfinite(h_vals) & (h_vals > 0)
                if mask.any():
                    plt.plot(t_vals[mask], h_vals[mask], 'o-', color='#e74c3c', linewidth=2, markersize=6)
                    plt.yscale('log')
                else:
                    plt.text(0.5, 0.5, 'No valid data', transform=plt.gca().transAxes, ha='center', fontsize=12, color='gray')
                plt.xlabel('Time (hours)')
                plt.ylabel('Hazard Rate h(t)')
                plt.title(f"{title}\n$R(t) = {result['R'][:100]}...$\n$h(t) = {result['h'][:100]}...$", fontsize=9)
                plt.grid(True, which='both', ls=':', alpha=0.7)
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                result['plot_base64'] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
            except Exception as e:
                result['plot_base64'] = None

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("FLASK SERVER HIDUP! http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)