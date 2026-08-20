"""
7. gün YouTube izlenme tahmini — eğitim scripti
================================================
Girdi : egitim_verisi tablosundan alınmış CSV (456 satır)
Çıktı : model.pkl
        ozellik_onemi.csv
        capraz_dogrulama_tahminleri.csv

Kurulum:
    pip install pandas numpy scikit-learn lightgbm joblib

Çalıştırma:
    python egitim.py egitim_verisi.csv

Yöntem özeti
------------
* Hedef doğrudan izlenme değil, log(7.gün / 24.saat) yani ÇARPAN.
  Böylece 500 izlenmeli video ile 500.000 izlenmeli video aynı ölçekte
  öğrenilir ve model asla negatif tahmin üretemez.
* artis_* kolonları saatlik ARTIŞ olduğu için kümülatif eğri burada
  cumsum ile yeniden kuruluyor.
* 456 satır az olduğundan ağaçlar sığ tutuldu ve çapraz doğrulama
  3 kez tekrarlanıp ortalaması alınıyor (tek bölmenin şansı skoru
  ±2-3 puan oynatabiliyor).
"""

import sys
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RepeatedKFold

warnings.filterwarnings("ignore", category=UserWarning)

CSV_YOLU = sys.argv[1] if len(sys.argv) > 1 else "egitim_verisi.csv"
HEDEF = "goruntulenme_7gun"
SAATLER = list(range(1, 25))
TEKRAR = 3          # çapraz doğrulama kaç kez tekrarlansın
KAT_SAYISI = 5
TOHUM = 42


# ======================================================================
# 0. ARTIŞ KOLONLARINI NORMALLEŞTİR
# ======================================================================
def artislari_normalize_et(df: pd.DataFrame) -> pd.DataFrame:
    """artis_* kolonları iki formatta gelebilir:
      (a) saatlik artış  — o saatte kaç izlenme geldi
      (b) kümülatif      — o saate kadar toplam kaç izlenme birikti
    Script (a) bekliyor. Format otomatik tespit edilip gerekirse
    kümülatiften saatlik artışa çevrilir.
    """
    df = df.copy()

    # eksik saat kolonu varsa oluştur
    for h in SAATLER:
        for tur in ("goruntulenme", "begeni", "yorum"):
            kolon = f"artis_{tur}_{h}saat"
            if kolon not in df.columns:
                print(f"  uyarı: {kolon} yok, 0 kabul edildi")
                df[kolon] = 0.0

    goruntulenme_kolonlari = [f"artis_goruntulenme_{h}saat" for h in SAATLER]
    v = df[goruntulenme_kolonlari].values.astype(float)

    # Tespit: satır boyunca değerler hiç düşmüyorsa kümülatiftir.
    # Saatlik artışta sayılar iner çıkar; kümülatifte hep artar.
    tam_satir = ~np.isnan(v).any(axis=1)
    if tam_satir.sum() == 0:
        kumulatif = False
    else:
        monoton = (np.diff(v[tam_satir], axis=1) >= 0).all(axis=1)
        kumulatif = monoton.mean() > 0.80

    for tur in ("goruntulenme", "begeni", "yorum"):
        kolonlar = [f"artis_{tur}_{h}saat" for h in SAATLER]
        if kumulatif:
            # Boşluk = ölçüm alınamadı; kümülatifte bir önceki değerle doldur
            m = df[kolonlar].ffill(axis=1).fillna(0).values.astype(float)
            m = np.maximum.accumulate(m, axis=1)          # düşüşleri düzelt
            df[kolonlar] = np.diff(m, axis=1, prepend=0.0)  # saatlik artışa çevir
        else:
            df[kolonlar] = df[kolonlar].fillna(0).clip(lower=0)

    if kumulatif:
        print("  format: artis_* kolonları KÜMÜLATİF → saatlik artışa çevrildi")
    else:
        print("  format: artis_* kolonları saatlik ARTIŞ, dönüşüm gerekmedi")

    return df


# ======================================================================
# 1. VERİYİ OKU VE TEMİZLE
# ======================================================================
def veriyi_oku(yol: str) -> pd.DataFrame:
    df = pd.read_csv(yol)
    print(f"Ham veri: {len(df)} satır, {df.shape[1]} kolon")

    df = artislari_normalize_et(df)

    baslangic = len(df)
    elenen = {}

    # --- hedefi olmayan satırlar eğitimde kullanılamaz ---
    once = len(df)
    df = df[df[HEDEF].notna() & (df[HEDEF] > 0)]
    elenen["hedef boş veya sıfır"] = once - len(df)

    # --- 24 saatlik kümülatif izlenme ---
    df["goruntulenme_24saat"] = df[
        [f"artis_goruntulenme_{h}saat" for h in SAATLER]
    ].sum(axis=1)

    # --- neredeyse hiç izlenmemiş videolar gürültü üretir ---
    once = len(df)
    df = df[df["goruntulenme_24saat"] >= 10]
    elenen["24 saatte 10'dan az izlenme"] = once - len(df)

    # --- 7. gün < 24. saat olamaz: veri toplama hatası ---
    once = len(df)
    df = df[df[HEDEF] >= df["goruntulenme_24saat"]]
    elenen["7. gün < 24. saat (tutarsız)"] = once - len(df)

    # --- absürt çarpanlar (viral olmuş ya da bozuk kayıt) ---
    carpan = df[HEDEF] / df["goruntulenme_24saat"].clip(lower=1)
    once = len(df)
    df = df[carpan <= 50]
    elenen["çarpan > 50x (aykırı)"] = once - len(df)

    for sebep, adet in elenen.items():
        if adet:
            print(f"  elendi — {sebep}: {adet}")
    print(f"Temizlik sonrası: {len(df)} satır "
          f"(toplam {baslangic - len(df)} satır elendi)")

    if len(df) < KAT_SAYISI * 4:
        print(f"\nDUR: sadece {len(df)} satır kaldı, eğitim anlamsız.")
        print("Yukarıdaki eleme sebeplerine bak — büyük ihtimalle kolon")
        print("formatı beklenenden farklı, veriyi silmeden önce onu düzelt.")
        sys.exit(1)

    if len(df) < 100:
        print("\nUYARI: 100'ün altında satır kaldı. Sonuçlara güvenme.")

    return df.reset_index(drop=True)


# ======================================================================
# 2. ÖZELLİK ÜRETİMİ
# ======================================================================
def ozellik_uret(df: pd.DataFrame):
    """CSV'deki ham kolonlardan model özelliklerini üretir.

    Döner: (X, v24)  —  X özellik tablosu, v24 24. saat kümülatif izlenme
    """
    X = pd.DataFrame(index=df.index)

    goruntulenme = df[[f"artis_goruntulenme_{h}saat" for h in SAATLER]].values.astype(float)
    begeni = df[[f"artis_begeni_{h}saat" for h in SAATLER]].values.astype(float)
    yorum = df[[f"artis_yorum_{h}saat" for h in SAATLER]].values.astype(float)

    kum = np.cumsum(goruntulenme, axis=1)      # kümülatif izlenme eğrisi
    v24 = kum[:, 23]
    guvenli_v24 = np.maximum(v24, 1.0)

    # ---------- kümülatif kontrol noktaları ----------
    for h in (1, 3, 6, 12, 18, 24):
        X[f"log_kum_{h}s"] = np.log1p(kum[:, h - 1])

    # ---------- büyüme oranları: eğrinin dikliği ----------
    def log_oran(ust, alt):
        return np.log1p(kum[:, ust - 1]) - np.log1p(kum[:, alt - 1])

    X["oran_3_1"] = log_oran(3, 1)
    X["oran_6_3"] = log_oran(6, 3)
    X["oran_12_6"] = log_oran(12, 6)
    X["oran_24_12"] = log_oran(24, 12)
    X["oran_24_6"] = log_oran(24, 6)

    # ---------- MOMENTUM ----------
    # Genelde en güçlü sinyal grubu. Video 24. saatte hâlâ hızlanıyorsa
    # 7. güne kadar çok daha uzağa gider.
    X["pay_ilk_3s"] = goruntulenme[:, 0:3].sum(axis=1) / guvenli_v24
    X["pay_son_6s"] = goruntulenme[:, 18:24].sum(axis=1) / guvenli_v24
    X["pay_son_12s"] = goruntulenme[:, 12:24].sum(axis=1) / guvenli_v24

    ilk6_ort = np.maximum(goruntulenme[:, 0:6].mean(axis=1), 0.5)
    son6_ort = goruntulenme[:, 18:24].mean(axis=1)
    X["ivme"] = np.log1p(son6_ort) - np.log1p(ilk6_ort)

    # son 6 saatteki doğrusal eğim (saat başına artış)
    saat_ekseni = np.arange(6, dtype=float)
    son6 = goruntulenme[:, 18:24]
    merkezli = saat_ekseni - saat_ekseni.mean()
    egim = (merkezli * (son6 - son6.mean(axis=1, keepdims=True))).sum(axis=1) / (merkezli ** 2).sum()
    X["son6_egim"] = np.sign(egim) * np.log1p(np.abs(egim))

    # ---------- eğrinin şekli ----------
    X["zirve_saati"] = goruntulenme.argmax(axis=1) + 1
    X["zirve_payi"] = goruntulenme.max(axis=1) / guvenli_v24
    X["dagilim"] = goruntulenme.std(axis=1) / np.maximum(goruntulenme.mean(axis=1), 0.5)
    X["yariya_ulasma_saati"] = (kum < (v24[:, None] / 2)).sum(axis=1) + 1
    X["sifir_saat_sayisi"] = (goruntulenme == 0).sum(axis=1)

    # ---------- etkileşim ----------
    toplam_begeni = begeni.sum(axis=1)
    toplam_yorum = yorum.sum(axis=1)
    X["log_begeni_24s"] = np.log1p(toplam_begeni)
    X["log_yorum_24s"] = np.log1p(toplam_yorum)
    X["begeni_oran"] = toplam_begeni / guvenli_v24
    X["yorum_oran"] = toplam_yorum / guvenli_v24
    X["yorum_begeni_oran"] = toplam_yorum / np.maximum(toplam_begeni, 1)
    X["begeni_momentum"] = begeni[:, 18:24].sum(axis=1) / np.maximum(toplam_begeni, 1)

    # ---------- kanal bağlamı ----------
    abone = df["abone_sayisi"].astype(float)
    kanal_ort = df["kanal_24saat_ortalama"].astype(float)

    X["log_abone"] = np.log1p(abone)
    X["log_kanal_yasi"] = np.log1p(df["kanal_yasi_gun"].astype(float))
    X["kanala_gore_oran"] = df["kanala_gore_oran_24saat"].astype(float)
    X["log_kanal_ortalama"] = np.log1p(kanal_ort)
    # video kanalın normalinin kaç katı — kanal büyüklüğünden bağımsız sinyal
    X["kanal_ustu_performans"] = np.log1p(v24) - np.log1p(kanal_ort)
    X["abone_penetrasyonu"] = v24 / np.maximum(abone, 1)

    # Boşluğun kendisi bilgi: bu videolar kanalın ilk videoları.
    # 456 satırın 44'ü böyle; silmek yerine bayrakla işaretliyoruz.
    X["kanal_gecmisi_yok"] = kanal_ort.isna().astype(int)

    # ---------- video özellikleri ----------
    sure = df["sure_saniye"].astype(float)
    X["log_sure"] = np.log1p(sure)
    X["kisa_video"] = (sure <= 60).astype(int)
    X["etiket_sayisi"] = df["etiket_sayisi"].astype(float)
    X["baslik_karakter"] = df["baslik_karakter_sayisi"].astype(float)
    X["baslik_kelime"] = df["baslik_kelime_sayisi"].astype(float)
    X["baslikta_sayi"] = df["baslikta_sayi_var_mi"].astype(float)
    X["baslikta_soru"] = df["baslikta_soru_var_mi"].astype(float)
    X["baslikta_unlem"] = df["baslikta_unlem_var_mi"].astype(float)

    # ---------- zaman ----------
    saat = pd.to_numeric(df["yayin_saati"], errors="coerce").astype(float)
    X["saat_sin"] = np.sin(2 * np.pi * saat / 24)
    X["saat_cos"] = np.cos(2 * np.pi * saat / 24)

    GUN_KODU = {
        "pazartesi": 0, "salı": 1, "sali": 1, "çarşamba": 2, "carsamba": 2,
        "perşembe": 3, "persembe": 3, "cuma": 4, "cumartesi": 5, "pazar": 6,
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    gun_ham = df["yayin_gunu"]
    if pd.api.types.is_numeric_dtype(gun_ham):
        gun = pd.to_numeric(gun_ham, errors="coerce").fillna(-1).astype(int)
    else:
        gun = (gun_ham.astype(str).str.strip().str.lower()
               .map(GUN_KODU).fillna(-1).astype(int))
        if (gun == -1).all():
            print("  uyarı: yayin_gunu değerleri tanınmadı, kolon etkisiz")
    X["yayin_gunu"] = gun
    X["hafta_sonu"] = gun.isin([5, 6]).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)
    return X, v24


# ======================================================================
# 3. DEĞERLENDİRME
# ======================================================================
def olc(gercek, tahmin, etiket):
    gercek = np.asarray(gercek, dtype=float)
    tahmin = np.asarray(tahmin, dtype=float)
    ape = np.abs(tahmin - gercek) / np.maximum(gercek, 1) * 100
    log_rmse = np.sqrt(np.mean((np.log1p(tahmin) - np.log1p(gercek)) ** 2))

    print(f"\n{etiket}")
    print(f"  MAPE (ortalama mutlak % hata) : {ape.mean():7.1f}%")
    print(f"  Medyan hata                   : {np.median(ape):7.1f}%")
    print(f"  ±%20 içinde kalan             : {(ape <= 20).mean() * 100:7.1f}%")
    print(f"  ±%50 içinde kalan             : {(ape <= 50).mean() * 100:7.1f}%")
    print(f"  log ölçekte RMSE              : {log_rmse:7.4f}")
    return ape


# ======================================================================
# 4. EĞİTİM
# ======================================================================
def egit(df: pd.DataFrame):
    X, v24 = ozellik_uret(df)
    y_ham = df[HEDEF].values.astype(float)
    guvenli_v24 = np.maximum(v24, 1.0)

    print(f"\nÖzellik sayısı: {X.shape[1]} | Satır: {len(X)}")

    # ---------- hedef dönüşümü: çarpanın logaritması ----------
    carpan = y_ham / guvenli_v24
    y = np.log(carpan)

    print("\nÇarpan dağılımı (7.gün / 24.saat):")
    print(f"  medyan {np.median(carpan):5.2f}x | "
          f"%10 {np.percentile(carpan, 10):5.2f}x | "
          f"%90 {np.percentile(carpan, 90):5.2f}x | "
          f"maks {carpan.max():5.2f}x")

    # ---------- çapraz doğrulama stratejisi ----------
    # Aynı kanalın videoları hem eğitimde hem testte olursa model kanalı
    # ezberler ve skor gerçekte olduğundan iyi çıkar. kanal_id varsa
    # GroupKFold ile bunu engelliyoruz.
    if "kanal_id" in df.columns and df["kanal_id"].notna().any():
        kanal_sayisi = df["kanal_id"].nunique()
        katlar = list(GroupKFold(n_splits=min(KAT_SAYISI, kanal_sayisi))
                      .split(X, y, groups=df["kanal_id"]))
        tekrar_sayisi = 1
        print(f"\nBölme: GroupKFold — {kanal_sayisi} kanal, kanal sızıntısı yok")
    else:
        katlar = list(RepeatedKFold(n_splits=KAT_SAYISI, n_repeats=TEKRAR,
                                    random_state=TOHUM).split(X))
        tekrar_sayisi = TEKRAR
        print(f"\nBölme: RepeatedKFold ({KAT_SAYISI} kat × {TEKRAR} tekrar)")
        print("  NOT: kanal_id kolonunu eklersen GroupKFold'a geçer ve skor")
        print("       gerçekçi olur. Şu anki skor iyimser olabilir.")

    # ---------- 456 satıra göre muhafazakâr parametreler ----------
    params = dict(
        objective="regression",
        metric="l2",
        learning_rate=0.02,
        num_leaves=7,              # sığ ağaç: bu boyutta 15 ezberler
        max_depth=4,
        min_child_samples=20,      # yaprak başına en az 20 örnek
        feature_fraction=0.6,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=0.5,
        lambda_l2=3.0,
        n_estimators=3000,
        verbose=-1,
        random_state=TOHUM,
    )

    oof_toplam = np.zeros(len(df))
    onem_toplam = np.zeros(X.shape[1])
    modeller = []
    turlar = []

    for tr, va in katlar:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_toplam[va] += model.predict(X.iloc[va])
        onem_toplam += model.feature_importances_
        modeller.append(model)
        turlar.append(model.best_iteration_ or params["n_estimators"])

    # Her satır her tekrarda tam bir kez doğrulama setine düştü
    oof = oof_toplam / tekrar_sayisi
    print(f"\nOrtalama ağaç sayısı: {int(np.mean(turlar))}")

    # ---------- log-çarpanı gerçek izlenmeye çevir ----------
    tahmin = np.exp(oof) * guvenli_v24

    # ---------- karşılaştırma tabanı ----------
    # Herkese aynı medyan çarpanı uygulayan aptal tahminci.
    # Model bunu belirgin geçemiyorsa sorun modelde değil veride.
    taban = np.median(carpan) * guvenli_v24

    olc(y_ham, taban, "TABAN — sabit medyan çarpan")
    ape_model = olc(y_ham, tahmin, "MODEL — LightGBM")

    taban_mape = (np.abs(taban - y_ham) / np.maximum(y_ham, 1) * 100).mean()
    iyilesme = (taban_mape - ape_model.mean()) / taban_mape * 100
    print(f"\nModelin tabana göre kazancı: {iyilesme:.1f}%")
    if iyilesme < 10:
        print("  → Kazanç düşük. Özellikler 7. günü açıklamıyor olabilir;")
        print("    daha fazla veri toplamak parametre ayarından daha değerli.")

    # ---------- özellik önemleri ----------
    onem_df = (pd.DataFrame({"ozellik": X.columns, "onem": onem_toplam})
               .sort_values("onem", ascending=False)
               .reset_index(drop=True))
    onem_df["pay_yuzde"] = (onem_df["onem"] / onem_df["onem"].sum() * 100).round(2)
    print("\nEn etkili 15 özellik:")
    print(onem_df.head(15).to_string(index=False))
    onem_df.to_csv("ozellik_onemi.csv", index=False)

    kullanilmayan = onem_df[onem_df["onem"] == 0]["ozellik"].tolist()
    if kullanilmayan:
        print(f"\nHiç kullanılmayan {len(kullanilmayan)} özellik: "
              f"{', '.join(kullanilmayan[:10])}")

    # ---------- satır bazında sonuçlar ----------
    sonuc = pd.DataFrame({
        "video_id": df["video_id"],
        "goruntulenme_24saat": v24.round().astype(int),
        "tahmin": tahmin.round().astype(int),
        "gercek": y_ham.round().astype(int),
        "hata_yuzde": ((tahmin - y_ham) / np.maximum(y_ham, 1) * 100).round(2),
    })
    sonuc.to_csv("capraz_dogrulama_tahminleri.csv", index=False)

    print("\nEn kötü 5 tahmin:")
    en_kotu = sonuc.reindex(
        sonuc["hata_yuzde"].abs().sort_values(ascending=False).index
    ).head(5)
    print(en_kotu.to_string(index=False))

    # ---------- modeli kaydet ----------
    joblib.dump(
        {"modeller": modeller,
         "kolonlar": list(X.columns),
         "medyan_carpan": float(np.median(carpan)),
         "surum": "v1"},
        "model.pkl",
    )
    print("\nKaydedildi: model.pkl, ozellik_onemi.csv, "
          "capraz_dogrulama_tahminleri.csv")
    return sonuc


# ======================================================================
# 5. YENİ VİDEOLAR İÇİN TAHMİN
# ======================================================================
def tahmin_et(df_yeni: pd.DataFrame, model_yolu: str = "model.pkl") -> np.ndarray:
    """Eğitimdekiyle aynı kolonlara sahip yeni satırlar için 7. gün tahmini.
    Tüm katların ortalaması alınır (topluluk tahmini)."""
    paket = joblib.load(model_yolu)

    df_yeni = artislari_normalize_et(df_yeni)

    X, v24 = ozellik_uret(df_yeni)
    X = X.reindex(columns=paket["kolonlar"])  # kolon sırasını sabitle

    log_carpan = np.mean([m.predict(X) for m in paket["modeller"]], axis=0)
    return (np.exp(log_carpan) * np.maximum(v24, 1)).round().astype(int)


if __name__ == "__main__":
    egit(veriyi_oku(CSV_YOLU))