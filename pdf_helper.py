from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import io, qrcode


def append_signature_page(pdf_file, signature_b64, original_hash_hex: str = None):
    pdf_file.seek(0)
    original_bytes = pdf_file.read()

    reader = PdfReader(io.BytesIO(original_bytes))
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    # 1. Generate QR Code
    qr = qrcode.make(signature_b64)
    img_byte_arr = io.BytesIO()
    qr.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # 2. Buat halaman signature
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 750, "Verifikasi Tanda Tangan Digital")

    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "Dokumen ini telah ditandatangani secara digital.")
    c.drawString(100, 710, "Metode: RSA (Kunci 2048-bit) & SHA256")

    c.drawImage(ImageReader(img_byte_arr), 100, 450, width=200, height=200)

    c.drawString(100, 420, "Pindai QR Code di atas untuk mendapatkan detail tanda tangan digital.")
    c.drawString(100, 390, "Detail Tanda Tangan (base64):")

    c.setFont("Courier", 8)
    max_chars_per_line = 80
    signature_lines = [
        signature_b64[i : i + max_chars_per_line]
        for i in range(0, len(signature_b64), max_chars_per_line)
    ]

    y_pos = 370
    for line in signature_lines:
        c.drawString(100, y_pos, line)
        y_pos -= 15

    c.save()
    packet.seek(0)

    writer.add_page(PdfReader(packet).pages[0])

    # 3. ✅ Simpan original_hash_hex di metadata PDF
    if original_hash_hex:
        writer.add_metadata({
            "/OriginalHash": original_hash_hex
        })

    output_pdf = io.BytesIO()
    writer.write(output_pdf)
    output_pdf.seek(0)

    return output_pdf


def get_original_hash_from_pdf(pdf_data: bytes) -> str | None:
    """Baca original hash dari metadata PDF bertanda tangan."""
    try:
        reader = PdfReader(io.BytesIO(pdf_data))
        metadata = reader.metadata or {}
        return metadata.get("/OriginalHash")
    except Exception as e:
        print(f"[ERROR] Gagal baca metadata: {e}")
        return None


def get_original_page_count(pdf_file):
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    return len(reader.pages)


def strip_last_page(pdf_data: bytes) -> bytes:
    """Buang halaman terakhir (halaman QR signature) dari PDF."""
    reader = PdfReader(io.BytesIO(pdf_data))
    writer = PdfWriter()
    for page in reader.pages[:-1]:
        writer.add_page(page)
    output = io.BytesIO()
    writer.write(output)
    output.seek(0)
    return output.read()