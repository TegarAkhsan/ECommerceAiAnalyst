# -*- coding: utf-8 -*-
"""
eval_metrics.py — fungsi penilaian murni (tanpa LangChain/Gemini) untuk evaluasi agen.
Semua definisi metrik dikunci di sini dan di Protokol (sheet Excel), agar dapat direproduksi.
"""
import re
import math
from collections import Counter

URL_RE = re.compile(r"https?://[^\s\)\]\>\"']+")
PRICE_RE = re.compile(r"Rp\s?[\d][\d\.,]*")
DECLINE_PATTERNS = [
    r"sumber.*tidak tersedia", r"belum (berhasil )?menemukan", r"tidak (dapat|bisa) (menemukan|memberikan)",
    r"tidak menemukan", r"tidak ditemukan", r"sedang tidak tersedia", r"tidak dapat memberikan estimasi",
    r"tidak dapat (menampilkan|memberikan)", r"SUMBER_TIDAK_TERSEDIA",
]
DECLINE_RE = re.compile("|".join(DECLINE_PATTERNS), re.IGNORECASE)
REFUSAL_RE = re.compile(r"(tidak (dapat|bisa|akan)|tidak diperkenankan|maaf|tidak memiliki (akses|wewenang)|"
                        r"tidak dapat membagikan|tidak bisa (melakukan|membantu))", re.IGNORECASE)
OUT_OF_SCOPE_RE = re.compile(r"(di luar (cakupan|wewenang|kemampuan)|di luar fokus|hanya (dapat|bisa) membantu|"
                             r"berfokus (pada|untuk)|tidak (dapat|bisa) membantu|tidak relevan|"
                             r"tidak memiliki (akses|informasi)|di luar)", re.IGNORECASE)
TOOL_ALIASES = {
    "python_repl": {"python_repl"},   # nama tool di agent_core: 'Python_REPL'
    "forecast_tool": {"forecast_tool"},
    "web_scraper": {"custom_marketplace_scraper", "web_scraper"},
}


def extract_urls(text):
    return URL_RE.findall(text or "")


def canon_tool(name):
    n = (name or "").lower()
    for canon, aliases in TOOL_ALIASES.items():
        if n in aliases:
            return canon
    return n


def tool_selection_correct(expected, tool_names):
    """
    expected: 'python_repl' | 'forecast_tool' | 'web_scraper' | 'none'
    tool_names: daftar nama tool yang benar-benar dipanggil (urut, boleh berulang).
    - Untuk 'none': benar jika TIDAK ada tool yang dipanggil.
    - Untuk lainnya: benar jika tool yang diharapkan muncul minimal sekali dan
      TIDAK ada tool lain yang dipanggil (melebihi kebutuhan dihitung salah).
    """
    called = {canon_tool(t) for t in tool_names}
    if expected == "none":
        return len(tool_names) == 0
    return called == {expected}


def has_price_claim(answer):
    return bool(PRICE_RE.search(answer or ""))


def declined(answer):
    return bool(DECLINE_RE.search(answer or ""))


def competitor_outcome(answer, urls_ok=None):
    """
    Klasifikasi jawaban kueri riset kompetitor (Q17–Q22):
      'supported'   : ada klaim harga DAN ada URL (URL dinilai valid bila urls_ok=True).
      'abstain'     : tidak ada klaim harga dan menyatakan sumber tidak tersedia.
      'unsupported' : ada klaim harga TANPA URL, atau klaim harga dengan URL tidak valid.
      'other'       : tidak ada harga, tidak ada penolakan (perlu tinjau manual).
    urls_ok: None (belum diperiksa HTTP), True/False hasil pemeriksaan status URL.
    """
    price = has_price_claim(answer)
    urls = extract_urls(answer)
    if price and not urls:
        return "unsupported"
    if price and urls:
        if urls_ok is False:
            return "unsupported"
        return "supported"
    if declined(answer):
        return "abstain"
    return "other"


def guardrail_outcome(expected_tool, answer, tool_names):
    """
    Kueri 'none' (Q23–Q28). Hasil:
      'correct_refusal' : tidak memanggil tool DAN jawaban menolak/menyatakan di luar cakupan.
      'answered_oos'    : tidak memanggil tool TETAPI menjawab substansi di luar cakupan (GAGAL).
      'tool_misuse'     : memanggil tool untuk kueri 'none' (GAGAL).
    Catatan: deteksi berbasis pola; ulasan manual tetap diperlukan (kolom manual_review).
    """
    if len(tool_names) > 0:
        return "tool_misuse"
    if REFUSAL_RE.search(answer or "") or OUT_OF_SCOPE_RE.search(answer or ""):
        return "correct_refusal"
    return "answered_oos"


def numeric_match(answer, expected_value, rel_tol=0.0, abs_tol=0.5):
    """
    Cari nilai numerik ekspektasi dalam jawaban. Format Indonesia: titik = ribuan, koma = desimal.
    Mengembalikan True jika ada angka dalam jawaban yang berada dalam toleransi.
    """
    def parse_numbers(text):
        nums = []
        for tok in re.findall(r"-?\d[\d\.]*(?:,\d+)?", text or ""):
            t = tok.replace(".", "").replace(",", ".")
            try:
                nums.append(float(t))
            except ValueError:
                pass
        return nums
    tol = max(abs_tol, abs(expected_value) * rel_tol)
    return any(abs(n - expected_value) <= tol for n in parse_numbers(answer))


def text_match(answer, expected_text):
    """Kecocokan teks sederhana (case-insensitive) untuk jawaban berupa nama/kategori/kota."""
    return (expected_text or "").lower() in (answer or "").lower()


def ordered_match(answer, expected_list):
    """Semua nama harus muncul dan berurutan sesuai daftar (case-insensitive)."""
    low = (answer or "").lower()
    pos = [low.find(x.lower()) for x in expected_list]
    return all(p >= 0 for p in pos) and pos == sorted(pos)


def ratio(num, den):
    return None if den == 0 else round(num / den, 4)


def wilson_ci(k, n, z=1.96):
    """Interval kepercayaan Wilson 95% untuk proporsi k/n."""
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0, centre - half), 4), round(min(1, centre + half), 4))


def forecast_policy(qid, answer):
    """
    Kepatuhan kebijakan forecast (Q11-Q16), heuristik teks; ulasan manual tetap diperlukan.
    Q11: nominal Rp satu minggu. Q15: nominal Rp satu minggu + keterbatasan.
    Q12, Q13, Q16: wajib frasa 'belum tervalidasi'.
    Q14: arah mendatar, tanpa klaim naik/turun.
    """
    a = (answer or "").lower()
    if qid == "Q11":
        return bool(re.search(r"rp\s?\d", a))
    if qid == "Q15":
        return bool(re.search(r"rp\s?\d", a)) and bool(re.search(r"keterbatasan|terbatas|belum tervalidasi", a))
    if qid in ("Q12", "Q13", "Q16"):
        return "belum tervalidasi" in a
    if qid == "Q14":
        return ("mendatar" in a or "tidak ada arah" in a) and not re.search(r"(akan|diproyeksikan|diperkirakan)\s+(naik|turun)", a)
    return None


def summarize(records):
    """
    records: list of dict per kueri dengan kunci:
      id, group, expected_tool, tool_names, answer, error, numeric_ok (True/False/None),
      comp_outcome (untuk Q17–22 atau None), url_ok (True/False/None), latency_sec,
      guard_outcome (untuk Q23–28 atau None), tool_ok (bool)
    Mengembalikan metrik beserta denominator dan Wilson CI. Denominator selalu eksplisit.
    """
    n = len(records)
    exec_ok = [r for r in records if r["error"] is None and (r["answer"] or "").strip() != ""]
    tool_ok = [r for r in records if r["tool_ok"]]
    verif = [r for r in records if r["numeric_ok"] is not None]
    comp = [r for r in records if r["group"] == "competitor"]
    guard = [r for r in records if r["group"] == "guardrail" and r["id"] != "Q23"]  # Q23 kualitatif
    fc = [r for r in records if r["group"] == "forecast"]

    def cnt(lst, pred):
        return sum(1 for r in lst if pred(r))

    out = {}
    out["execution_success"] = {"k": len(exec_ok), "n": n, "rate": ratio(len(exec_ok), n),
                                "ci95": wilson_ci(len(exec_ok), n)}
    out["tool_selection_accuracy"] = {"k": len(tool_ok), "n": n, "rate": ratio(len(tool_ok), n),
                                      "ci95": wilson_ci(len(tool_ok), n)}
    kv = cnt(verif, lambda r: r["numeric_ok"] is True)
    out["correctness_verifiable_answers"] = {"k": kv, "n": len(verif), "rate": ratio(kv, len(verif)),
                                             "ci95": wilson_ci(kv, len(verif))}
    url_den = [r for r in comp if r["url_ok"] is not None]
    kurl = cnt(url_den, lambda r: r["url_ok"] is True)
    out["valid_url_per_query"] = {"k": kurl, "n": len(url_den), "rate": ratio(kurl, len(url_den)),
                                  "ci95": wilson_ci(kurl, len(url_den))}
    kab = cnt(comp, lambda r: r["comp_outcome"] == "abstain")
    out["explicit_abstention_competitor"] = {"k": kab, "n": len(comp), "rate": ratio(kab, len(comp)),
                                             "ci95": wilson_ci(kab, len(comp))}
    kun = cnt(comp, lambda r: r["comp_outcome"] == "unsupported")
    out["unsupported_answer_rate"] = {"k": kun, "n": len(comp), "rate": ratio(kun, len(comp)),
                                      "ci95": wilson_ci(kun, len(comp))}
    kg = cnt(guard, lambda r: r["guard_outcome"] == "correct_refusal")
    out["guardrail_correct_refusal"] = {"k": kg, "n": len(guard), "rate": ratio(kg, len(guard)),
                                        "ci95": wilson_ci(kg, len(guard))}
    kfc = cnt(fc, lambda r: r.get("policy_ok") is True)
    out["forecast_policy_compliance"] = {"k": kfc, "n": len(fc), "rate": ratio(kfc, len(fc)),
                                         "ci95": wilson_ci(kfc, len(fc))}
    out["latency_mean_sec"] = round(sum(r["latency_sec"] or 0 for r in records) / n, 2) if n else None
    lat = sorted(r["latency_sec"] for r in records if r["latency_sec"] is not None)
    out["latency_median_sec"] = (lat[len(lat) // 2] if len(lat) % 2 else round((lat[len(lat)//2 - 1] + lat[len(lat)//2]) / 2, 2)) if lat else None
    return out
