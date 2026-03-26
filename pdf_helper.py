from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import io, qrcode


def append_signature_page(pdf_file, signature_b64):
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    writer = PdfWriter()

    # Salin semua halaman dari PDF asli
    for page in reader.pages:
        writer.add_page(page)

    # 1. Generate QR Code
    qr = qrcode.make(signature_b64)
    img_byte_arr = io.BytesIO()
    qr.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # 2. Buat Kertas Baru (Mengikuti Desain Colab Anda)
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 750, "Verifikasi Tanda Tangan Digital")

    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "Dokumen ini telah ditandatangani secara digital.")
    c.drawString(100, 710, "Metode: RSA (Kunci 2048-bit) & SHA256")

    # Tampilkan QR Code
    c.drawImage(ImageReader(img_byte_arr), 100, 450, width=200, height=200)

    c.drawString(
        100,
        420,
        "Pindai QR Code di atas untuk mendapatkan detail tanda tangan digital.",
    )
    c.drawString(100, 390, "Detail Tanda Tangan (base64):")

    # Pecah tanda tangan menggunakan logika Colab Anda
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

    # 3. Gabungkan PDF Asli dan Kertas Baru
    writer.add_page(PdfReader(packet).pages[0])

    output_pdf = io.BytesIO()
    writer.write(output_pdf)
    output_pdf.seek(0)

    return output_pdf
