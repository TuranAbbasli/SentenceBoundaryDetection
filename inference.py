import time

from azcharboundary.segmenter import TextSegmenter

def run_case(segmenter: TextSegmenter, text: str, case_name: str = "Testing!"):
    """
    Uses inference function make prediction on given text

    Args:
        segmenter (TextSegmenter): Model which will make prediction
        text (str): Text on which predictions will be made
        case_name (str): Case name for logging
    
    
    """
    print(f"\n=== Running case: {case_name} ===")
    start = time.time()

    output = segmenter.inference(text=text, threshold=0.5)

    end = time.time()
    inference_time_ms = (end - start) * 1000

    print(f"--- Inference time: {inference_time_ms:.2f} ms ---")
    print("--- Segmentation results ---")
    print(output)


def test_inference(model_dir: str):
    """Inference test over multiple cases"""

    segmenter = TextSegmenter()
    segmenter.load(model_dir)

    # CASE 1 — Legal domain
    text_legal = (
        "Azərbaycan Respublikası Konstitusiyasının 32-ci maddəsinə əsasən, "
        "hər kəsin şəxsi və ailə həyatına hörmət hüququ vardır. "
        "Heç kəs şəxsi məlumatlarının qanunsuz toplanmasına və yayılmasına məruz qala bilməz. "
        "Məhkəmə qərarı olmadan şəxsin telefon danışıqlarına nəzarət edilməsi qadağandır."
    )
    run_case(segmenter, text_legal, "Legal domain")

    # CASE 2 — General text
    text_general = (
        "Bu gün hava çox gözəldir. Səhər tezdən külək əsirdi, amma indi sakitdir. "
        "Axşam yağış yağacağı proqnozlaşdırılır."
    )
    run_case(segmenter, text_general, "General domain")

    # CASE 3 — Long paragraph (stress test)
    text_long = (
        "Azərbaycan iqtisadiyyatı son illərdə sürətli inkişaf edir. "
        "Bu inkişaf müxtəlif sahələrdə özünü göstərir. "
        "Xüsusilə texnologiya, təhsil və enerji sektorunda ciddi dəyişikliklər var. "
        "Bir çox startaplar yaranır, dövlət innovasiyalara investisiya edir. "
        "Bu proses ölkənin rəqəmsal transformasiyasını daha da gücləndirir."
    )
    run_case(segmenter, text_long, "Long text (stress test)")

    # CASE 4 — Edge case: Short text
    text_short = "Salam dünya."
    run_case(segmenter, text_short, "Short text")

    # CASE 5 — Edge case: No sentence-ending punctuation
    text_no_punct = "Bu test cümləsi heç bir nöqtə işarəsi yoxdur və model bununla necə işləyəcək görək"
    run_case(segmenter, text_no_punct, "No punctuation")

    # CASE 6 — Edge case: Many punctuation marks
    text_punct = "Bu nədir?! Siz bunu gördünüzmü?! Yox, inanmıram..."
    run_case(segmenter, text_punct, "Heavy punctuation")

if __name__ == "__main__":
    model_path = "azcharboundary/models/model_v1.xz"
    test_inference(model_dir=model_path)
