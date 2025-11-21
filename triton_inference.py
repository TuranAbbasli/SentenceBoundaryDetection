import time
from typing import List, Tuple

import numpy as np
import tritonclient.http as httpclient

from azcharboundary.utils.constants import TERMINAL_SENTENCE_CHAR_LIST, SENTENCE_TAG
from azcharboundary.utils.features import FeatureExtractor

# Triton HTTP endpoint and model name for the FIL model
TRITON_URL: str = "localhost:8000"
MODEL_NAME: str = "model"

# Single global feature extractor instance reused across calls
feature_extractor = FeatureExtractor()


def preprocessing(
    text: str,
    left_window: int = 5,
    right_window: int = 5,
) -> Tuple[np.ndarray, List[int]]:
    """
    Client-side preprocessing step.

    - Scans the input text for terminal sentence characters.
    - Extracts features for characters at those positions using FeatureExtractor.
    - Returns a feature matrix and the corresponding character indices.

    Args:
        text: Input text chunk.
        left_window: Number of characters to the left for the feature window.
        right_window: Number of characters to the right for the feature window.

    Returns:
        A tuple (features, positions):
          - features: np.ndarray of shape (N, 19) or (N, D) before reshaping, dtype float32.
          - positions: list of indices (length N) of terminal characters in `text`.
    """
    terminal_indices: List[int] = []

    # Pre-identify all terminal characters to batch-process them
    for i, char in enumerate(text):
        if char in TERMINAL_SENTENCE_CHAR_LIST:
            terminal_indices.append(i)

    if not terminal_indices:
        # No terminals -> empty feature matrix
        return np.empty((0, 19), dtype=np.float32), terminal_indices

    terminal_features = feature_extractor.get_char_features(
        text,
        left_window,
        right_window,
        positions=terminal_indices,
    )

    features_np = np.asarray(terminal_features, dtype=np.float32)

    # Ensure features are 2D (N, 19). If your FeatureExtractor already returns (N, 19),
    # this will be a no-op shape-wise.
    if features_np.ndim == 1:
        features_np = features_np.reshape(-1, 19)

    return features_np, terminal_indices


def postprocessing(
    text: str,
    labels: List[int],
    positions: List[int],
) -> str:
    """
    Client-side postprocessing step.

    - Takes the original text, a label per terminal position, and the terminal positions.
    - Inserts SENTENCE_TAG after each character whose label == 1.

    Args:
        text: Original text chunk.
        labels: List of 0/1 labels (same length as positions).
        positions: List of indices of terminal characters in the original text.

    Returns:
        A single string with SENTENCE_TAG tokens inserted after predicted boundaries.
    """
    if not labels or not positions:
        # Nothing to insert
        return text

    result_chars: List[str] = list(text)

    tag_len = len(SENTENCE_TAG)
    tag_shift = 0  # how many extra characters we've inserted so far

    for prediction, terminal_idx in zip(labels, positions):
        if prediction:
            insert_pos = terminal_idx + 1 + tag_shift
            result_chars.insert(insert_pos, SENTENCE_TAG)
            tag_shift += tag_len

    return "".join(result_chars)


def run_case(
    client: httpclient.InferenceServerClient,
    text: str,
    case_name: str = "Testing!",
) -> float:
    """
    Run a single test case through Triton + local pre/postprocessing and log timing.

    Pipeline:
      1. Preprocess text locally -> (features, positions)
      2. If features are non-empty:
           - Call Triton FIL model with FEATURES
           - Convert probabilities to labels
      3. Postprocess locally -> insert SENTENCE_TAG at predicted boundaries
      4. Print input, output, and total inference time.

    Args:
        client: Triton HTTP client instance.
        text: Input text chunk.
        case_name: Human-readable name for the test case.

    Returns:
        Inference time in milliseconds for this case (pre + Triton + post).
    """
    print(f"\n=== Running case: {case_name} ===")

    # 1. Local preprocessing
    start = time.time()
    features, positions = preprocessing(text)

    if features.shape[0] == 0:
        # No terminal characters: skip Triton, just return original text
        output_text = text
        end = time.time()
        inference_time_ms = (end - start) * 1000.0
    else:
        # 2. Call Triton FIL model with FEATURES
        inputs: List[httpclient.InferInput] = []
        outputs: List[httpclient.InferRequestedOutput] = []

        # Triton FIL model expects input name "input__0" with shape [N, 19], type FP32
        n_rows, n_cols = features.shape
        inp = httpclient.InferInput("input__0", [n_rows, n_cols], "FP32")
        inp.set_data_from_numpy(features)

        # Request the probability output from the FIL model
        out = httpclient.InferRequestedOutput("output__0")

        inputs.append(inp)
        outputs.append(out)

        response = client.infer(MODEL_NAME, inputs=inputs, outputs=outputs)
        end = time.time()
        inference_time_ms = (end - start) * 1000.0

        probs = response.as_numpy("output__0")  # shape (N, 2), dtype float32

        # probs[:, 0] = p(class 0), probs[:, 1] = p(class 1)
        labels_np = (probs[:, 1] > probs[:, 0]).astype(np.int32)
        labels: List[int] = labels_np.tolist()

        # 3. Local postprocessing
        output_text = postprocessing(text, labels, positions)

    print(f"--- Inference time: {inference_time_ms:.2f} ms ---")
    print("--- Segmentation results ---")
    print(f"Input: {text}\n")
    print(f"Output: {output_text}\n")

    return inference_time_ms


def test_inference() -> float:
    """
    Run a suite of test cases against the Triton FIL model,
    using local preprocessing and postprocessing.

    Returns:
        Average inference time in milliseconds across all test cases.
    """
    client = httpclient.InferenceServerClient(url=TRITON_URL)

    # Optional sanity checks
    if not client.is_server_live():
        raise RuntimeError("Triton server is not live")
    if not client.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model '{MODEL_NAME}' is not ready on Triton")

    print("Connected to Triton. Model is ready.")

    inference_times_ms: List[float] = []

    # CASE 1 — Legal domain
    text_legal = (
        "Azərbaycan Respublikası Konstitusiyasının 32-ci maddəsinə əsasən, "
        "hər kəsin şəxsi və ailə həyatına hörmət hüququ vardır. "
        "Heç kəs şəxsi məlumatlarının qanunsuz toplanmasına və yayılmasına məruz qala bilməz. "
        "Məhkəmə qərarı olmadan şəxsin telefon danışıqlarına nəzarət edilməsi qadağandır."
    )
    inference_times_ms.append(run_case(client, text_legal, "Legal domain"))

    # CASE 2 — General text
    text_general = (
        "Bu gün hava çox gözəldir. Səhər tezdən külək əsirdi, amma indi sakitdir. "
        "Axşam yağış yağacağı proqnozlaşdırılır."
    )
    inference_times_ms.append(run_case(client, text_general, "General domain"))

    # CASE 3 — Long paragraph (stress test)
    text_long = (
        "Azərbaycan iqtisadiyyatı son illərdə sürətli inkişaf edir. "
        "Bu inkişaf müxtəlif sahələrdə özünü göstərir. "
        "Xüsusilə texnologiya, təhsil və enerji sektorunda ciddi dəyişikliklər var. "
        "Bir çox startaplar yaranır, dövlət innovasiyalara investisiya edir. "
        "Bu proses ölkənin rəqəmsal transformasiyasını daha da gücləndirir."
    )
    inference_times_ms.append(run_case(client, text_long, "Long text (stress test)"))

    # CASE 4 — Edge case: Short text
    text_short = "Salam dünya."
    inference_times_ms.append(run_case(client, text_short, "Short text"))

    # CASE 5 — Edge case: Many punctuation marks
    text_punct = "Bu nədir?! Siz bunu gördünüzmü?! Yox, inanmıram..."
    inference_times_ms.append(run_case(client, text_punct, "Heavy punctuation"))

    # CASE 6 — Legal: list of short phrases (clauses, fragments)
    text_legal_phrases = (
        "a) Müqavilənin ləğvi; b) tərəflərin razılığı; c) məhkəmə qərarı; "
        "ç) müflislik elan edilməsi; d) qanunvericiliyin dəyişməsi; "
        "e) əhəmiyyətli şərtlərin pozulması; ə) gecikmiş öhdəlik; "
        "f) tərəflərin məsuliyyəti."
    )
    inference_times_ms.append(
        run_case(client, text_legal_phrases, "Legal domain — list of phrases")
    )

    # CASE 7 — Legal: list of sentences separated by yeni sətr/bullet-like struktur
    text_legal_list_sentences = (
        "1) Müqavilə yalnız yazılı formada bağlandıqda etibarlı sayılır. "
        "2) Tərəflər müqavilə üzrə öhdəliklərini vicdanla yerinə yetirməlidirlər. "
        "3) Mübahisələr danışıqlar yolu ilə həll edilmədikdə, məhkəməyə müraciət oluna bilər. "
        "4) Tərəflər arasında yaranan ziyan, qanunvericiliyə uyğun olaraq kompensasiya edilir. "
        "5) Müqavilənin müddəti bitdikdə, tərəflərin yazılı razılığı ilə uzadıla bilər."
    )
    inference_times_ms.append(
        run_case(
            client,
            text_legal_list_sentences,
            "Legal domain — numbered list of sentences",
        )
    )

    # CASE 8 — Legal: çox uzun cümlə (tək cümləlik stress test)
    text_legal_long_sentence = (
        "İddiaçı iddia ərizəsində göstərmişdir ki, cavabdeh tərəfindən "
        "müqavilə öhdəliklərinin vaxtında yerinə yetirilməməsi nəticəsində ona "
        "maddi ziyan dəymiş, bu ziyanın məbləği isə müstəqil auditor rəyi ilə "
        "təsdiq edilmişdir və həmin məbləğin, həmçinin gecikdirməyə görə hesablanmış "
        "dəbbə pulu və məhkəmə xərclərinin cavabdehdən tutulmasını xahiş etmişdir."
    )
    inference_times_ms.append(
        run_case(client, text_legal_long_sentence, "Legal domain — single long sentence")
    )

    # CASE 9 — Legal: abbreviations, maddə istinadları, rəqəmlər
    text_legal_abbrev = (
        "AR Mülki Məcəlləsinin 422.1-ci maddəsinə əsasən, müqavilə tərəflərinin "
        "öz öhdəliklərini lazımi qaydada yerinə yetirməsi məcburidir. "
        "Eyni Məcəllənin 439.2-ci maddəsinə görə, borclu öhdəliyi yerinə yetirmədikdə, "
        "kreditor ona qarşı məhkəməyə müraciət edə bilər. Bu halda 5 mlyn. cərimə tətbiq olur."
        "Bu Qanun və digər normativ-hüquqi aktlar (o cümlədən, \"İstehlakçıların hüquqlarının müdafiəsi haqqında\" Qanun) "
        "istehlakçıların mənafeyini qorumağa yönəlib."
    )
    inference_times_ms.append(
        run_case(client, text_legal_abbrev, "Legal domain — abbreviations & article refs")
    )

    # CASE 10 — Legal: sitatlar, mötərizələr, tirelər
    text_legal_quotes = (
        "Məhkəmə qərarında qeyd edilir ki, \"tərəflər arasında bağlanmış müqavilə "
        "bozucu şərt həyata keçənədək qüvvədə qalır\". "
        "Hakim belə nəticəyə gəlmişdir ki, cavabdehin hərəkətləri "
        "(öhdəliyin qəsdən yerinə yetirilməməsi və qarşı tərəfin ziyana salınması) "
        "qanunvericiliyin tələblərinə ziddir — bu halda, əlavə məsuliyyət tədbirləri tətbiq oluna bilər."
    )
    inference_times_ms.append(
        run_case(client, text_legal_quotes, "Legal domain — quotes & parentheses")
    )

    # CASE 11 — Legal: tarixlər, faizlər, qarışıq struktur
    text_legal_mixed = (
        "2019-cu il 15 mart tarixli kredit müqaviləsinə əsasən, borc məbləği "
        "50 000 (əlli min) manat müəyyən edilmiş, illik faiz dərəcəsi isə 18% olmuşdur. "
        "Müqavilənin 7.2-ci bəndinə görə, borcalan 30 (otuz) gün ərzində ödənişi "
        "etmədikdə, bank gecikdirilmiş hər günə görə əlavə 0.1% dəbbə pulu hesablayır. "
        "Tərəflər arasında bu müddəa ilə bağlı hər hansı yazılı etiraz qeydə alınmamışdır."
    )
    inference_times_ms.append(
        run_case(client, text_legal_mixed, "Legal domain — dates, numbers, percentages")
    )

    # CASE 12 — Legal: çoxlu qısa cümlələr, müxtəlif nöqtələnmə
    text_legal_many_short = (
        "İddia rədd edilir. Apellyasiya şikayəti təmin olunmur. "
        "Qərar elan olundu. Tərəflərə izah edildi. "
        "Qərardan kassasiya qaydasında şikayət vermək hüququ saxlanılır!"
    )
    inference_times_ms.append(
        run_case(client, text_legal_many_short, "Legal domain — many short sentences")
    )

    # CASE 13 — Legal: qarışıq dil (az + bəzi ingilis hüquqi terminləri)
    text_legal_mixed_lang = (
        "Bu müqavilə Azərbaycan Respublikası qanunvericiliyinə uyğun olaraq tənzimlənir. "
        "Hər hansı dispute tərəflər arasında negotiation yolu ilə həll edilmədikdə, "
        "mübahisə Bakı Kommersiya Məhkəməsində arbitration istisna olunmaqla baxılır. "
        "Force majeure hallarına təbii fəlakətlər, müharibə və hökumət qərarları daxildir."
    )
    inference_times_ms.append(
        run_case(
            client,
            text_legal_mixed_lang,
            "Legal domain — mixed language & terms",
        )
    )

    avg_inference_ms = sum(inference_times_ms) / len(inference_times_ms)
    print(
        "\nAverage inference time over {} cases: {:.2f} ms".format(
            len(inference_times_ms), avg_inference_ms
        )
    )
    return avg_inference_ms


if __name__ == "__main__":
    test_inference()
