"""
Karar Destek Sistemi (KDS) - AHP & TOPSIS Güven Endeksi Motoru
Çok kriterli risk değerlendirmesi (C1-C5), Güven Endeksi (C_i^*) hesaplaması
ve Pareto (80/20) kritik risk bölgesi (alt %20) raporlaması üretir.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd


# Uzman Odaklı AHP Kriter Ağırlıkları (CR = 0.0126 < 0.10)
# Sırasıyla: C1_MaliyetSapma, R2_UrunUretilememe, R3_ButceRevizesi,
#            R4_KaliteProblemi, R5_TerminRevizesi
DEFAULT_WEIGHTS = np.array([0.419, 0.061, 0.160, 0.097, 0.263], dtype=float)

CRITERIA_COLUMNS = [
    "C1_MaliyetSapma",
    "R2_UrunUretilememe",
    "R3_ButceRevizesi",
    "R4_KaliteProblemi",
    "R5_TerminRevizesi",
]

SAMPLE_DATA = """BuyerGrup\tC1_MaliyetSapma\tR2_UrunUretilememe\tR3_ButceRevizesi\tR4_KaliteProblemi\tR5_TerminRevizesi
KIRTASİYE\t41.755638\t0.00000072\t0.00000000\t0.00000358\t0.00000000
DENİZ\t107.504802\t0.00008858\t0.00000000\t0.00000000\t0.00000000
AMERİKAN SERVİS SUPLA\t89.265607\t0.00010179\t0.00000000\t0.00000000\t0.00000000
BİJUTERİ\t105.320014\t0.00001600\t0.00000107\t0.00000000\t0.00000107
DEKORATİF AKSESUAR\t123.918468\t0.00002034\t0.00000387\t0.00001937\t0.00000000
MUM\t461.555082\t0.00015855\t0.00000000\t0.00001865\t0.00000000
SEYAHAT\t462.558943\t0.00000527\t0.00000790\t0.00000263\t0.00000000
MUTFAK GRUBU\t1180.573311\t0.00006006\t0.00000000\t0.00000626\t0.00000000
SABUN\t1247.863159\t0.00000214\t0.00000000\t0.00000641\t0.00000000
ÇORAP\t785.439750\t0.00002469\t0.00000302\t0.00000224\t0.00000168
AKSESUAR\t1451.449605\t0.00001572\t0.00000786\t0.00000561\t0.00000000
AKTİF SPOR\t12046.426526\t0.00006945\t0.00002762\t0.00000908\t0.00000395
AYAKKABI\t12066.525943\t0.00004584\t0.00000805\t0.00000608\t0.00000523
BANYO\t38796.367201\t0.00005967\t0.00000000\t0.00000000\t0.00000000
BLAZER CEKET\t65346.938030\t0.00004119\t0.00000000\t0.00002060\t0.00000000
VALİZ\t73877.477130\t0.00003500\t0.00001200\t0.00001500\t0.00000800
"""


def load_decision_matrix(input_path: str | None = None) -> pd.DataFrame:
    """Dış dosyadan (CSV, TSV/TXT veya Excel) karar matrisini okur; verilmezse örnek veriyi yükler."""
    if input_path is None:
        df = pd.read_csv(io.StringIO(SAMPLE_DATA), sep="\t")
    else:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Girdi dosyası bulunamadı: {path}")

        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, sep=None, engine="python")

    missing_cols = [col for col in ["BuyerGrup", *CRITERIA_COLUMNS] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Eksik sütun(lar) bulundu: {missing_cols}")

    df = df.copy()
    for col in CRITERIA_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def run_topsis(
    df: pd.DataFrame,
    weights: np.ndarray = DEFAULT_WEIGHTS,
    critical_bottom_ratio: float = 0.20,
) -> pd.DataFrame:
    """
    Tüm kriterler minimizasyon (maliyet/risk) yönlü olacak şekilde TOPSIS
    Güven Endeksi (C_i^*) skorunu ve risk segmentini hesaplar.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(CRITERIA_COLUMNS),):
        raise ValueError(f"Ağırlık vektörü {len(CRITERIA_COLUMNS)} elemanlı olmalıdır.")

    # Ağırlık toplamını 1'e normalize et
    weights = weights / weights.sum()

    matrix = df[CRITERIA_COLUMNS].to_numpy(dtype=float)

    # Adım 1: Vektör Normalizasyonu (r_ij = x_ij / sqrt(sum(x_ij^2)))
    norms = np.sqrt((matrix ** 2).sum(axis=0))
    norms = np.where(norms == 0, 1.0, norms)
    norm_matrix = matrix / norms

    # Adım 2: Ağırlıklı Normalize Karar Matrisi (v_ij = r_ij * w_j)
    weighted_matrix = norm_matrix * weights

    # Adım 3: İdeal (A+, minimum risk) ve Negatif İdeal (A-, maksimum risk)
    ideal_solution = np.min(weighted_matrix, axis=0)
    negative_ideal_solution = np.max(weighted_matrix, axis=0)

    # Adım 4: Öklid Uzaklıkları (d_star, d_minus)
    d_star = np.sqrt(((weighted_matrix - ideal_solution) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted_matrix - negative_ideal_solution) ** 2).sum(axis=1))

    # Adım 5: Güven Endeksi (C_i^* = d_minus / (d_star + d_minus))
    denominator = d_star + d_minus
    c_star = np.where(denominator == 0, 1.0, d_minus / denominator)

    result = df.copy()
    result["Guven_Endeksi_Skoru(C*)"] = np.round(c_star, 6)
    result["Risk_Sıralaması"] = (
        result["Guven_Endeksi_Skoru(C*)"].rank(ascending=False, method="min").astype(int)
    )
    result = result.sort_values(
        by=["Guven_Endeksi_Skoru(C*)", "BuyerGrup"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # Alt %20 dilimi "Kritik Riskli Alan (Kırmızı Bölge)" olarak işaretle
    n_rows = len(result)
    n_critical = max(1, int(np.ceil(n_rows * critical_bottom_ratio)))
    result["Risk_Bölgesi"] = "Normal / İzleme"
    result.loc[n_rows - n_critical :, "Risk_Bölgesi"] = "Kritik Riskli Alan (Kırmızı Bölge)"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AHP-TOPSIS Karar Destek Sistemi Güven Endeksi Hesaplayıcı"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Karar matrisi dosyası (.csv, .tsv, .txt veya .xlsx). Verilmezse örnek veri kullanılır.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="KDS_Nihai_Karar_Raporu.xlsx",
        help="Çıktı Excel rapor dosyasının yolu (varsayılan: KDS_Nihai_Karar_Raporu.xlsx)",
    )
    args = parser.parse_args()

    df_input = load_decision_matrix(args.input)
    df_ranked = run_topsis(df_input)

    print("=== TOPSIS GÜVEN ENDEKSİ SIRALAMASI (İLK 5 - EN GÜVENİLİR) ===")
    cols_to_show = ["Risk_Sıralaması", "BuyerGrup", "C1_MaliyetSapma", "Guven_Endeksi_Skoru(C*)", "Risk_Bölgesi"]
    print(df_ranked[cols_to_show].head(5).to_string(index=False))

    print("\n=== KRİTİK RİSKLİ ALAN (ALT %20 - KIRMIZI BÖLGE) ===")
    critical_df = df_ranked[df_ranked["Risk_Bölgesi"].str.startswith("Kritik")]
    print(critical_df[cols_to_show].to_string(index=False))

    output_path = Path(args.output)
    df_ranked.to_excel(output_path, index=False)
    print(f"\nSistem Başarıyla Çalıştı: Güven Endeksi Raporu oluşturuldu -> {output_path}")


if __name__ == "__main__":
    main()