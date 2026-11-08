"""Receipt rendering.

Layouts mirror the real corpus: PH BIR forms place a vendor header above a
customer block above a money column; POS slips are a narrow single column. Label
wording and fonts vary per receipt so extraction is not measured against one fixed
phrasing - which would make the corpus easier than reality.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.eval.synthetic.spec import ReceiptSpec

FONT_DIRECTORIES = (
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/truetype/msttcorefonts"),
)
FONT_FALLBACKS = {
    "arial": ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
    "calibri": ("calibri.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
    "verdana": ("verdana.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
    "tahoma": ("tahoma.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
    "consola": ("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"),
    "cour": ("cour.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"),
    "lucon": ("lucon.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"),
    "times": ("times.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf"),
}

# Alternative wordings for the same concept.
TOTAL_LABELS = ("TOTAL AMOUNT DUE", "Total Amount Due", "TOTAL AMT DUE", "AMOUNT DUE")
NET_LABELS = ("VATable Sales", "Amount Net of VAT", "VATable Sales (Net of VAT)")
VAT_LABELS = ("VAT 12%", "Less: VAT", "Value Added Tax", "12% VAT")
SALES_LABELS = ("Total Sales (VAT Inclusive)", "Total Sales", "Gross Amount")
SERVICE_LABELS = ("Service Charge", "Svc Charge", "Service Chg")


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = FONT_FALLBACKS.get(name, ("arial.ttf",))
    for candidate in candidates:
        for directory in FONT_DIRECTORIES:
            path = directory / candidate
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size)
                except OSError:
                    continue
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class _Canvas:
    """Layout helper, so rendering code reads as a sequence of lines."""

    def __init__(self, width: int, height: int, font_name: str, base_size: int) -> None:
        self.image = Image.new("L", (width, height), color=250)
        self.draw = ImageDraw.Draw(self.image)
        self.width = width
        self.y = 24
        self.font_name = font_name
        self.base_size = base_size

    def font(self, delta: int = 0) -> ImageFont.FreeTypeFont:
        return load_font(self.font_name, max(9, self.base_size + delta))

    def line(self, text: str, delta: int = 0, indent: int = 24,
             centre: bool = False, gap: int = 6) -> None:
        font = self.font(delta)
        x = indent
        if centre:
            span = self.draw.textlength(text, font=font)
            x = max(indent, (self.width - span) / 2)
        self.draw.text((x, self.y), text, fill=25, font=font)
        self.y += max(9, self.base_size + delta) + gap

    def pair(self, label: str, value: str, delta: int = 0, indent: int = 24,
             gap: int = 6) -> None:
        """Label left, value right-aligned - the layout money actually lives in."""
        font = self.font(delta)
        self.draw.text((indent, self.y), label, fill=25, font=font)
        span = self.draw.textlength(value, font=font)
        self.draw.text((self.width - indent - span, self.y), value, fill=25, font=font)
        self.y += max(9, self.base_size + delta) + gap

    def rule(self, gap: int = 10) -> None:
        self.draw.line([(24, self.y), (self.width - 24, self.y)], fill=120, width=1)
        self.y += gap

    def finish(self) -> Image.Image:
        bottom = min(self.y + 24, self.image.height)
        return self.image.crop((0, 0, self.width, bottom))


def _money(value: float) -> str:
    return f"{value:,.2f}"


def render_receipt(spec: ReceiptSpec) -> Image.Image:
    if spec.template in ("ph_vat_or", "ph_nonvat_si"):
        return _render_form(spec)
    return _render_slip(spec)


def _render_form(spec: ReceiptSpec) -> Image.Image:
    rng = random.Random(spec.index + 7)
    canvas = _Canvas(1000, 1600, spec.font_name, 21)
    is_vat = spec.vat_classification == "vat"

    canvas.line(spec.vendor_name.upper(), delta=6, centre=True)
    for address in spec.address_lines:
        canvas.line(address, delta=-4, centre=True)

    registration = (f"VAT Reg. TIN {spec.vendor_tax_id}" if is_vat
                    else f"Non VAT Reg. TIN: {spec.vendor_tax_id}")
    if spec.proprietor:
        registration += f"  *  {spec.proprietor} - Prop."
    canvas.line(registration, delta=-3, centre=True)
    canvas.rule()

    canvas.line("OFFICIAL RECEIPT" if is_vat else "SALES INVOICE", delta=4, centre=True)
    canvas.pair(f"No. {spec.invoice_number}", f"Date: {spec.date_text}")
    canvas.rule()

    if spec.customer_name:
        canvas.line(f"Registered Name : {spec.customer_name}", delta=-3)
    if spec.customer_tax_id:
        canvas.line(f"TIN : {spec.customer_tax_id}", delta=-3)
    if spec.customer_name or spec.customer_tax_id:
        canvas.line("Business Address : 6023 Sacred Heart cor. Kamagong Sts.", delta=-4)
        canvas.rule()

    canvas.pair("Nature of Service / Item", "Qty      Amount", delta=-3)
    for item in spec.items:
        canvas.pair(item.name, f"{item.quantity}     {_money(item.amount)}", delta=-3)
        if item.description:
            # Indented and unpriced, as receipts print continuation lines.
            canvas.line(item.description, delta=-5, indent=48, gap=4)
    canvas.rule()

    if is_vat:
        canvas.pair(rng.choice(SALES_LABELS), _money(spec.total_sales))
        canvas.pair(rng.choice(NET_LABELS), _money(spec.net_sales))
        canvas.pair(rng.choice(VAT_LABELS), _money(spec.printed_tax_amount or 0.0))
        canvas.pair("Zero-Rated Sales", "0.00", delta=-4)
        canvas.pair("VAT-Exempt Sales (SC/PWD)", "0.00", delta=-4)
    else:
        canvas.pair("Total Sales", _money(spec.total_sales))

    if spec.service_charge:
        canvas.pair(rng.choice(SERVICE_LABELS), _money(spec.service_charge))

    canvas.rule()
    canvas.pair(rng.choice(TOTAL_LABELS), _money(spec.total_amount), delta=4)
    canvas.rule()

    if not is_vat:
        canvas.line("THIS DOCUMENT IS NOT VALID FOR CLAIMING INPUT TAXES", delta=-5)
    canvas.line(f"BIR Authority to Print No. {rng.randint(10**11, 10**12)}", delta=-6)
    canvas.line("Date of ATP: 02-20-2026    Valid until: 02-20-2031", delta=-6)
    canvas.line("100 Booklets  (50x2)  9001-14000", delta=-6)
    return canvas.finish()


def _render_slip(spec: ReceiptSpec) -> Image.Image:
    rng = random.Random(spec.index + 11)
    canvas = _Canvas(560, 1300, spec.font_name, 19)

    canvas.line(spec.vendor_name, delta=3, centre=True)
    for address in spec.address_lines:
        canvas.line(address, delta=-4, centre=True)

    if spec.country == "PH":
        canvas.line(f"VAT Reg. TIN {spec.vendor_tax_id}", delta=-4, centre=True)
    elif spec.country == "US":
        canvas.line(f"({rng.randint(200, 989)}) {rng.randint(200, 999)}-"
                    f"{rng.randint(1000, 9999)}", delta=-4, centre=True)
    else:
        canvas.line(f"Sdn. Bhd.  SST Reg. No. {rng.randint(10**9, 10**10)}",
                    delta=-4, centre=True)

    canvas.line("*" * 34, delta=-4, centre=True)
    canvas.line(spec.date_text, delta=-3)
    canvas.line(f"Invoice number: {spec.invoice_number}", delta=-3)
    canvas.line("*" * 34, delta=-4, centre=True)

    for item in spec.items:
        canvas.pair(f"{item.quantity} {item.name[:22]}", _money(item.amount), delta=-3)
        if item.description:
            canvas.line(item.description[:34], delta=-5, indent=40, gap=4)
    canvas.rule()

    if spec.country == "PH":
        canvas.pair("Total Sales", _money(spec.total_sales), delta=-1)
        canvas.pair("VATable Sales", _money(spec.net_sales), delta=-2)
        canvas.pair("VAT 12%", _money(spec.printed_tax_amount or 0.0), delta=-2)
    else:
        canvas.pair("Subtotal", _money(spec.net_sales), delta=-1)
        canvas.pair("Sales Tax" if spec.country == "US" else "SST 6%",
                    _money(spec.printed_tax_amount or 0.0), delta=-2)

    canvas.pair("TOTAL", _money(spec.total_amount), delta=2)
    canvas.pair("Payment", _money(round(spec.total_amount + 20, 2)), delta=-3)
    canvas.pair("Change Due", _money(20.0), delta=-3)
    canvas.line("*" * 34, delta=-4, centre=True)
    canvas.line("Thank you and please come again", delta=-5, centre=True)
    return canvas.finish()
