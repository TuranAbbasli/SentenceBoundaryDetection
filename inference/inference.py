import time

from azcharboundary.segmenter import TextSegmenter


def run_case(segmenter: TextSegmenter, text: str, case_name: str = "Testing!") -> float:
    """
    Uses inference function to make prediction on given text
    and returns inference time in milliseconds.
    """
    print(f"\n=== Running case: {case_name} ===")
    start = time.time()

    output = segmenter.inference(text=text, threshold=0.5)

    end = time.time()
    inference_time_ms = (end - start) * 1000

    print(f"--- Inference time: {inference_time_ms:.2f} ms ---")
    print("--- Segmentation results ---")
    print(f"Input: {text}\n")
    print(f"Output: {output}\n")

    return inference_time_ms


def test_inference(model_dir: str) -> float:
    """Inference test over multiple cases. Returns average inference time in ms."""
    segmenter = TextSegmenter()

    load_start = time.time()
    segmenter.load(model_dir)
    print("Model has been loaded! Time took: {:.2f} s".format(time.time() - load_start))

    inference_times_ms = []

    # CASE 1 — Legal contract, dates, abbreviations, mixed list styles
    text_case_1 = (
        "Bu Müqavilə 12.05.2015-ci il tarixdə Bakı şəh., AZ0000, Nizami k., 15 ünvanında "
        "“Alpha Consulting” MMC (bundan sonra — “Podratçı”) və “Beta Group” ASC "
        "(bundan sonra — “Müştəri”) arasında imzalanmışdır. "
        "Müqavilə AR Mülki Məcəlləsinin 422.1-ci maddəsinə, həmçinin AR Konstitusiyasının 32-ci maddəsinə uyğun "
        "olaraq tənzimlənir. "
        "Tərəflərin hüquq və vəzifələri aşağıdakı kimidir: "
        "a) Podratçı işləri 01.06.2015-dək tam həcmdə yerinə yetirməlidir; "
        "b) Müştəri 10 (on) bank günü ərzində ödənişi həyata keçirməlidir; "
        "c) Force majeure (təbii fəlakət, müharibə və s.) hallarında məsuliyyət istisna olunur; "
        "ç) tərəflər arasında yaranan dispute əvvəlcə negotiation yolu ilə həll edilir. "
        "Müqavilə müddəti bitdikdən sonra tərəflərin yazılı razılığı (e-mail, faks və ya imzalı məktub) "
        "olmadan avtomatik uzadılmış sayılmır."
    )
    inference_times_ms.append(run_case(segmenter, text_case_1, "Case 1 — Contract + lists + dates"))

    # CASE 2 — Court decision with numbered paragraphs, internal abbreviations and mixed punctuation
    text_case_2 = (
        "Bakı Apellyasiya Məhk. məhkəmə heyəti, hakim İ.Xəlilovun sədrliyi ilə açıq məhkəmə iclasında "
        "iddiaçı A.A.-nın cavabdeh “Gamma” MMC-yə qarşı iddiası üzrə işi baxaraq müəyyən etdi ki, "
        "mübahisə predmeti 15.09.2019-cu il tarixli satınalma müqaviləsinin icrası ilə bağlıdır. "
        "1) İddiaçı bildirmişdir ki, cavabdeh öhdəliyi vaxtında yerinə yetirməyib, nəticədə 5 000 (beş min) manat "
        "maddi ziyan dəymişdir; 2) Cavabdeh isə, öz növbəsində, öhdəliyin pozulmasını force majeure ilə əsaslandırmış, "
        "lakin bu barədə hər hansı rəsmi sübut (sertifikat, akt və s.) təqdim etməmişdir. "
        "Məhkəmə hesab edir ki, cavabdehin arqumentləri əsassızdır... "
        "Nəticə etibarilə, iddia qismən təmin edilir?! "
        "Qərar elan olundu və tərəflərə izah edildi ki, qərardan 1 (bir) ay müddətində kassasiya şikayəti verilə bilər."
    )
    inference_times_ms.append(run_case(segmenter, text_case_2, "Case 2 — Court decision, numbered items"))

    # CASE 3 — Mixed language, percentages, times, inline list items, tricky abbreviations
    text_case_3 = (
        "12.03.2020 tarixli kredit müqaviləsinə (№ KM-2020/03-12) əsasən, borc məbləği 75 000 (yetiş beş min) manat "
        "təyin edilmişdir. İllik faiz dərəcəsi 18,5% olaraq müəyyən edilib; gecikmə halında isə əlavə 0,1% dəbbə pulu "
        "hesablanır. "
        "Clause 5.2-də qeyd olunur: \"Borc veren shall provide audited financial statements\" — lakin azərbaycanca "
        "versiyada “audit edilmiş maliyyə hesabatı” ifadəsi istifadə olunmuşdur. "
        "Saat 10:30-da tərəflər bankın mərkəzi ofisində (Bakı ş., Heydər Əliyev pr., 10) görüşərək aşağıdakıları "
        "razılaşdırmışlar: (i) ödəniş qrafiki yenidən tərtib olunur; (ii) 3 (üç) ay müddətinə grace period tətbiq edilir; "
        "(iii) borcalanın əlavə təminat təqdim etməsi tələb edilmir. "
        "Bu protokol, Bank Nəzarəti Şöb., həmçinin Risk Dept. tərəfindən də təsdiq edilib."
    )
    inference_times_ms.append(run_case(segmenter, text_case_3, "Case 3 — Mixed language, %, times, lists"))

    # CASE 4 — Long narrative, quotes, parentheses, ellipses, heavy punctuation, fake sentence-like abbreviations
    text_case_4 = (
        "Məhkəmə iclasında cavabdeh belə demişdir: \"Mən müqaviləni oxumuşam, lakin oradakı ‘7.2-ci bənd’ "
        "mənə aydın olmayıb\". Hakim sual verir: \"Siz hüquqşünasla məsləhətləşmisinizmi?!\" "
        "Cavabdeh cavab verir ki, o, yalnız tanışı olan bir mütəxəssislə (prof. S.Əliyev) qısa müzakirə aparıb, "
        "lakin rəsmi legal opinion almamışdır. "
        "Zalda olan nümayəndə (şirk. nümay., yəni rəsmi təmsilçi) isə bildirir ki, tərəflər arasında "
        "‘gentlemen’s agreement’ də olub... Lakin bu, yazılı formada təsdiq edilməyib. "
        "Hakim qeyd edir ki, belə informal razılaşmalar AR qanunvericiliyində ayrıca təsbit olunmayıb, "
        "bu səbəbdən də məhk. onları hüquqi əsas kimi qəbul etmir. "
        "Bu halda, yalnız müqavilənin mətni, əlavə razılaşmalar (Annex 1, Annex 2 və s.) və tərəflərin faktiki davranışı "
        "nəzərə alınır."
    )
    inference_times_ms.append(run_case(segmenter, text_case_4, "Case 4 — Quotes, ellipses, fake endings"))

    # CASE 5 — Mixed obligations, bullets, broken structures, dates and no-punct fragments
    text_case_5 = (
        "“Delta Logistic” MMC ilə bağlanmış 05.11.2018-ci il tarixli xidmət müqaviləsinin 3-cü bölməsi "
        "öhdəliklərin icrasına həsr olunmuşdur. Bölmə aşağıdakı bəndlərdən ibarətdir: "
        "1) Podratçı yükün təhlükəsiz daşınmasını təmin etməlidir — yük itdikdə və ya zədələndikdə, "
        "Mülki Məcəllənin 921-ci maddəsinə uyğun olaraq məsuliyyət daşıyır; "
        "2) Müştəri xidmət haqqını 30 (otuz) təqvim günü ərzində ödəməlidir; "
        "3) tərəflər aşağıdakı hallarda müqaviləni birtərəfli qaydada ləğv edə bilərlər: "
        "a) müflislik elan edilməsi; b) 60 gündən artıq gecikmə; c) qanunvericiliyin dəyişməsi nəticəsində "
        "müqavilənin icrasının faktiki olaraq mümkünsüz olması. "
        "Bundan əlavə, qeydlər bölməsində belə yazılmışdır: "
        "“Əlavə xidmətlər göstərilə bilər qiymət sonradan razılaşdırılır hər bir tərəf bu barədə əvvəlcədən yazılı "
        "məlumat təqdim etməlidir” — cümlə ardıcıllığı pozulmuş, nöqtə və vergüllər isə, praktiki olaraq, "
        "heç yerdə qoyulmamışdır. "
        "Sonda 01.01.2019 tarixli əlavə razılaşma ilə qiymətlər 10% artırılmış, lakin əvvəlki qrafik dəyişdirilməmişdir."
    )
    inference_times_ms.append(run_case(segmenter, text_case_5, "Case 5 — Bullets, broken text, dates"))

    avg_inference_ms = sum(inference_times_ms) / len(inference_times_ms)
    print("\nAverage inference time over {} cases: {:.2f} ms".format(
        len(inference_times_ms), avg_inference_ms
    ))
    return avg_inference_ms


if __name__ == "__main__":
    model_path = "azcharboundary/trained_models/v3/checkpoint.tl"
    test_inference(model_dir=model_path)
