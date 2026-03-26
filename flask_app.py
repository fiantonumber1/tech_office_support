from flask import Flask, request, jsonify, send_from_directory, send_file
import sympy as sp
import pandas as pd
import re
import os
from werkzeug.utils import secure_filename
from signal_processing import compute_features, compute_fft_plot, compute_time_plot
from crypto_helper import hash_file_data, generate_rsa_keys, sign_data, verify_signature
from pdf_helper import append_signature_page, strip_last_page,get_original_hash_from_pdf
import io

app = Flask(__name__)

UPLOAD_DIR = "uploaded_files"
BASE_PUBLIC_URL = "http://147.93.103.168:5632/files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
API_KEY = "SUPER_SECRET_KEY"


## FFT - Signal Processing
@app.route('/fft', methods=['POST'])
def compute_fft():

    file = request.files['file']
    sampling_rate = float(request.form.get('sampling_rate', 1000))

    df = pd.read_excel(file)
    signal = df.iloc[:,1].dropna().astype(float).values

    features = compute_features(signal)

    # ✅ 2 gambar
    time_image = compute_time_plot(signal, sampling_rate)
    fft_image = compute_fft_plot(signal, sampling_rate)

    return jsonify({
        "time_image": time_image,
        "fft_image": fft_image,
        "features": features
    })


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

## public file

# =====================================================
# ROUTES — FILE UPLOAD (FOR LARAVEL)
# =====================================================
@app.route('/upload-public', methods=['POST'])
def upload_file():
    if request.headers.get("X-API-KEY") != API_KEY:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"ok": False, "error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)

    try:
        file.save(save_path)
        return jsonify({
            "ok": True,
            "url": f"{BASE_PUBLIC_URL}/{filename}"
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# =====================================================
# ROUTES — PUBLIC FILE ACCESS
# =====================================================
@app.route('/files/<path:filename>', methods=['GET'])
def serve_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)





@app.route('/generate-keys', methods=['GET'])
def generate_keys_route():
    priv_pem, pub_pem = generate_rsa_keys()
    return jsonify({
        "status": "success",
        "private_key": priv_pem,
        "public_key": pub_pem
    })
 
 
@app.route('/sign', methods=['POST'])
def sign_pdf_route():
    if 'file' not in request.files or 'private_key' not in request.form:
        return jsonify({"error": "File PDF dan private_key wajib dikirim!"}), 400
 
    pdf_file = request.files['file']
    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Format file ditolak! Hanya menerima dokumen PDF."}), 400
 
    # ✅ Pertahankan nama file asli, fallback ke 'signed_document' kalau gagal
    try:
        raw_name = pdf_file.filename or "signed_document"
        original_filename = os.path.splitext(secure_filename(raw_name))[0] or "signed_document"
        download_filename = f"{original_filename}_signed.pdf"
    except Exception:
        download_filename = "signed_document.pdf"

    raw_private_key = request.form['private_key'].replace('\\n', '\n')
    private_key_pem = raw_private_key.encode('utf-8')
 
    pdf_data = pdf_file.read()
    file_hash = hash_file_data(pdf_data)
    original_hash_hex = file_hash.hex()
 
    try:
        signature_b64 = sign_data(file_hash, private_key_pem)
        print("\n" + "="*50)
        print("👉 COPY SIGNATURE INI UNTUK VERIFIKASI:")
        print(signature_b64)
        print(f"👉 ORIGINAL HASH HEX: {original_hash_hex}")
        print("="*50 + "\n")
    except Exception as e:
        print(f"[ERROR SIGNING] Kunci salah atau rusak: {str(e)}")
        return jsonify({"error": "Private Key tidak valid atau format rusak!"}), 400
 
    output_pdf = append_signature_page(
        io.BytesIO(pdf_data),
        signature_b64,
        original_hash_hex=original_hash_hex
    )
 
    print(f"[DEBUG] download_filename: {download_filename}")
    response = send_file(output_pdf, mimetype='application/pdf', as_attachment=True, download_name=download_filename)
    response.headers['X-Signature'] = signature_b64
    return response

@app.route('/verify', methods=['POST'])
def verify_pdf_route():
    if 'file' not in request.files or 'signature' not in request.form or 'public_key' not in request.form:
        return jsonify({"error": "Data tidak lengkap! Butuh file, signature, dan public_key."}), 400
 
    pdf_file = request.files['file']
    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Hanya menerima dokumen PDF untuk dicek."}), 400
 
    raw_public_key = request.form['public_key'].replace('\\n', '\n')
    public_key_pem = raw_public_key.encode('utf-8')
    signature_b64 = request.form['signature'].strip()
 
    pdf_data = pdf_file.read()
 
    # ✅ Baca original hash dari metadata PDF (bukan re-hash, bukan strip)
    original_hash_hex = get_original_hash_from_pdf(pdf_data)
    if not original_hash_hex:
        return jsonify({
            "status": "invalid",
            "message": "PALSU! Metadata hash tidak ditemukan. Dokumen bukan hasil signing sistem ini."
        })
 
    try:
        original_hash_bytes = bytes.fromhex(original_hash_hex)
    except Exception:
        return jsonify({"status": "invalid", "message": "PALSU! Format hash di metadata rusak."})
 
    try:
        verify_signature(signature_b64, original_hash_bytes, public_key_pem)
        return jsonify({"status": "valid", "message": "Dokumen ASLI dan tidak ada perubahan."})
    except Exception as e:
        print(f"[ERROR VERIFY] Percobaan gagal: {str(e)}")
        return jsonify({"status": "invalid", "message": "PALSU! Dokumen telah diedit atau kunci salah."})

# =====================================================
# MAIN
# =====================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5632, debug=True)
