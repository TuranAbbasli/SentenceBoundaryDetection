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

    # CASE 1 — Legal domain
    text_legal = (
        "Azərbaycan Respublikası Konstitusiyasının 32-ci maddəsinə əsasən, "
        "hər kəsin şəxsi və ailə həyatına hörmət hüququ vardır. "
        "Heç kəs şəxsi məlumatlarının qanunsuz toplanmasına və yayılmasına məruz qala bilməz. "
        "Məhkəmə qərarı olmadan şəxsin telefon danışıqlarına nəzarət edilməsi qadağandır."
    )
    inference_times_ms.append(run_case(segmenter, text_legal, "Legal domain"))

    # CASE 2 — General text
    text_general = (
        "Bu gün hava çox gözəldir. Səhər tezdən külək əsirdi, amma indi sakitdir. "
        "Axşam yağış yağacağı proqnozlaşdırılır."
    )
    inference_times_ms.append(run_case(segmenter, text_general, "General domain"))

    # CASE 3 — Long paragraph (stress test)
    text_long = (
        "Azərbaycan iqtisadiyyatı son illərdə sürətli inkişaf edir. "
        "Bu inkişaf müxtəlif sahələrdə özünü göstərir. "
        "Xüsusilə texnologiya, təhsil və enerji sektorunda ciddi dəyişikliklər var. "
        "Bir çox startaplar yaranır, dövlət innovasiyalara investisiya edir. "
        "Bu proses ölkənin rəqəmsal transformasiyasını daha da gücləndirir."
    )
    inference_times_ms.append(run_case(segmenter, text_long, "Long text (stress test)"))

    # CASE 4 — Edge case: Short text
    text_short = "Salam dünya."
    inference_times_ms.append(run_case(segmenter, text_short, "Short text"))

    # CASE 5 — Edge case: No sentence-ending punctuation
    text_no_punct = (
        "Bu test cümləsi heç bir nöqtə işarəsi yoxdur və model bununla necə işləyəcək görək"
    )
    inference_times_ms.append(run_case(segmenter, text_no_punct, "No punctuation"))

    # CASE 6 — Edge case: Many punctuation marks
    text_punct = "Bu nədir?! Siz bunu gördünüzmü?! Yox, inanmıram..."
    inference_times_ms.append(run_case(segmenter, text_punct, "Heavy punctuation"))

    # CASE 7 — Legal: list of short phrases (clauses, fragments)
    text_legal_phrases = (
        "a) Müqavilənin ləğvi; b) tərəflərin razılığı; c) məhkəmə qərarı; "
        "ç) müflislik elan edilməsi; d) qanunvericiliyin dəyişməsi; "
        "e) əhəmiyyətli şərtlərin pozulması; ə) gecikmiş öhdəlik; "
        "f) tərəflərin məsuliyyəti."
    )
    inference_times_ms.append(
        run_case(segmenter, text_legal_phrases, "Legal domain — list of phrases")
    )

    # CASE 8 — Legal: list of sentences separated by yeni sətr/bullet-like struktur
    text_legal_list_sentences = (
        "1) Müqavilə yalnız yazılı formada bağlandıqda etibarlı sayılır. "
        "2) Tərəflər müqavilə üzrə öhdəliklərini vicdanla yerinə yetirməlidirlər. "
        "3) Mübahisələr danışıqlar yolu ilə həll edilmədikdə, məhkəməyə müraciət oluna bilər. "
        "4) Tərəflər arasında yaranan ziyan, qanunvericiliyə uyğun olaraq kompensasiya edilir. "
        "5) Müqavilənin müddəti bitdikdə, tərəflərin yazılı razılığı ilə uzadıla bilər."
    )
    inference_times_ms.append(
        run_case(
            segmenter,
            text_legal_list_sentences,
            "Legal domain — numbered list of sentences",
        )
    )

    # CASE 9 — Legal: çox uzun cümlə (tək cümləlik stress test)
    text_legal_long_sentence = (
        "İddiaçı iddia ərizəsində göstərmişdir ki, cavabdeh tərəfindən "
        "müqavilə öhdəliklərinin vaxtında yerinə yetirilməməsi nəticəsində ona "
        "maddi ziyan dəymiş, bu ziyanın məbləği isə müstəqil auditor rəyi ilə "
        "təsdiq edilmişdir və həmin məbləğin, həmçinin gecikdirməyə görə hesablanmış "
        "dəbbə pulu və məhkəmə xərclərinin cavabdehdən tutulmasını xahiş etmişdir."
    )
    inference_times_ms.append(
        run_case(segmenter, text_legal_long_sentence, "Legal domain — single long sentence")
    )

    # CASE 10 — Legal: abbreviations, maddə istinadları, rəqəmlər
    text_legal_abbrev = (
        "AR Mülki Məcəlləsinin 422.1-ci maddəsinə əsasən, müqavilə tərəflərinin "
        "öz öhdəliklərini lazımi qaydada yerinə yetirməsi məcburidir. "
        "Eyni Məcəllənin 439.2-ci maddəsinə görə, borclu öhdəliyi yerinə yetirmədikdə, "
        "kreditor ona qarşı məhkəməyə müraciət edə bilər. Bu halda 5 mlyn. cərimə tətbiq olur."
        "Bu Qanun və digər normativ-hüquqi aktlar (o cümlədən, \"İstehlakçıların hüquqlarının müdafiəsi haqqında\" Qanun) "
        "istehlakçıların mənafeyini qorumağa yönəlib."
    )
    inference_times_ms.append(
        run_case(segmenter, text_legal_abbrev, "Legal domain — abbreviations & article refs")
    )

    # CASE 11 — Legal: sitatlar, mötərizələr, tirelər
    text_legal_quotes = (
        "Məhkəmə qərarında qeyd edilir ki, \"tərəflər arasında bağlanmış müqavilə "
        "bozucu şərt həyata keçənədək qüvvədə qalır\". "
        "Hakim belə nəticəyə gəlmişdir ki, cavabdehin hərəkətləri "
        "(öhdəliyin qəsdən yerinə yetirilməməsi və qarşı tərəfin ziyana salınması) "
        "qanunvericiliyin tələblərinə ziddir — bu halda, əlavə məsuliyyət tədbirləri tətbiq oluna bilər."
    )
    inference_times_ms.append(
        run_case(segmenter, text_legal_quotes, "Legal domain — quotes & parentheses")
    )

    # CASE 12 — Legal: tarixlər, faizlər, qarışıq struktur
    text_legal_mixed = (
        "2019-cu il 15 mart tarixli kredit müqaviləsinə əsasən, borc məbləği "
        "50 000 (əlli min) manat müəyyən edilmiş, illik faiz dərəcəsi isə 18% olmuşdur. "
        "Müqavilənin 7.2-ci bəndinə görə, borcalan 30 (otuz) gün ərzində ödənişi "
        "etmədikdə, bank gecikdirilmiş hər günə görə əlavə 0.1% dəbbə pulu hesablayır. "
        "Tərəflər arasında bu müddəa ilə bağlı hər hansı yazılı etiraz qeydə alınmamışdır."
    )
    inference_times_ms.append(
        run_case(segmenter, text_legal_mixed, "Legal domain — dates, numbers, percentages")
    )

    # CASE 13 — Legal: çoxlu qısa cümlələr, müxtəlif nöqtələnmə
    text_legal_many_short = (
        "İddia rədd edilir. Apellyasiya şikayəti təmin olunmur. "
        "Qərar elan olundu. Tərəflərə izah edildi. "
        "Qərardan kassasiya qaydasında şikayət vermək hüququ saxlanılır!"
    )
    inference_times_ms.append(
        run_case(segmenter, text_legal_many_short, "Legal domain — many short sentences")
    )

    # CASE 14 — Legal: qarışıq dil (az + bəzi ingilis hüquqi terminləri)
    text_legal_mixed_lang = (
        "Bu müqavilə Azərbaycan Respublikası qanunvericiliyinə uyğun olaraq tənzimlənir. "
        "Hər hansı dispute tərəflər arasında negotiation yolu ilə həll edilmədikdə, "
        "mübahisə Bakı Kommersiya Məhkəməsində arbitration istisna olunmaqla baxılır. "
        "Force majeure hallarına təbii fəlakətlər, müharibə və hökumət qərarları daxildir."
    )
    inference_times_ms.append(
        run_case(
            segmenter,
            text_legal_mixed_lang,
            "Legal domain — mixed language & terms",
        )
    )

    avg_inference_ms = sum(inference_times_ms) / len(inference_times_ms)
    print("\nAverage inference time over {} cases: {:.2f} ms".format(
        len(inference_times_ms), avg_inference_ms
    ))
    return avg_inference_ms


if __name__ == "__main__":
    model_path = "azcharboundary/models/model_v1.tl"
    test_inference(model_dir=model_path)
