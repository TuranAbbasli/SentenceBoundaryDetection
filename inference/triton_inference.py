import time
from typing import List, Optional, Tuple, Dict

import numpy as np
import tritonclient.grpc as grpcclient

from azcharboundary.utils.constants import TERMINAL_SENTENCE_CHAR_LIST, SENTENCE_TAG
from azcharboundary.utils.features import FeatureExtractor


# Triton gRPC endpoint and model name for the FIL model
TRITON_URL: str = "localhost:8001"
MODEL_NAME: str = "model"

# Connection configuration
NETWORK_TIMEOUT = 60.0  # seconds

# Model I/O configuration
INPUT_NAME = "input__0"
OUTPUT_NAME = "output__0"
FEATURE_DIM = 19

# Single global feature extractor instance reused across calls
feature_extractor = FeatureExtractor()


def preprocessing(
    text: str,
    left_window: int = 5,
    right_window: int = 5,
) -> Tuple[Optional[List[grpcclient.InferInput]],
           Optional[List[grpcclient.InferRequestedOutput]],
           List[int]]:
    """
    Preprocessing step.

    Responsibilities:
      - Scan the input text for terminal sentence characters.
      - Extract features for those positions.
      - Build Triton InferInput and InferRequestedOutput objects.

    Args:
        text: Input text chunk.
        left_window: Number of characters to the left for the feature window.
        right_window: Number of characters to the right for the feature window.

    Returns:
        (inputs, outputs, positions):
          - inputs: list of InferInput for Triton, or None if there is nothing to send.
          - outputs: list of InferRequestedOutput, or None if there is nothing to send.
          - positions: list of indices of terminal characters in `text`.
    """
    terminal_indices: List[int] = [
        i for i, char in enumerate(text) if char in TERMINAL_SENTENCE_CHAR_LIST
    ]

    if not terminal_indices:
        # No terminal characters: nothing to send to Triton
        return None, None, []

    # Extract features for the detected terminal positions
    terminal_features = feature_extractor.get_char_features(
        text,
        left_window,
        right_window,
        positions=terminal_indices,
    )

    features_np = np.asarray(terminal_features, dtype=np.float32)

    # Ensure features are 2D (N, FEATURE_DIM)
    if features_np.ndim == 1:
        features_np = features_np.reshape(-1, FEATURE_DIM)

    if features_np.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected feature dimension {FEATURE_DIM}, got {features_np.shape[1]}"
        )

    n_rows, n_cols = features_np.shape

    # Build Triton input
    inp = grpcclient.InferInput(INPUT_NAME, [n_rows, n_cols], "FP32")
    inp.set_data_from_numpy(features_np)

    # Request probability output from the FIL model
    out = grpcclient.InferRequestedOutput(OUTPUT_NAME)

    return [inp], [out], terminal_indices


def postprocessing(
    text: str,
    positions: List[int],
    response: Optional[grpcclient.InferResult],
) -> str:
    """
    Postprocessing step.

    Responsibilities:
      - Convert Triton response probabilities into labels.
      - Insert SENTENCE_TAG after characters predicted as sentence boundaries.

    Args:
        text: Original text chunk.
        positions: List of indices of terminal characters in the original text.
        response: Triton InferResult (or None if no inference was performed).

    Returns:
        Text with SENTENCE_TAG tokens inserted after predicted boundaries.
    """
    if response is None or not positions:
        # Nothing to insert
        return text

    # Extract probabilities from Triton response
    probs = response.as_numpy(OUTPUT_NAME)
    if probs is None:
        raise RuntimeError(
            f"No output received from Triton for output name '{OUTPUT_NAME}'"
        )

    # probs[:, 0] = p(class 0), probs[:, 1] = p(class 1)
    labels_np = (probs[:, 1] > probs[:, 0]).astype(np.int32)
    labels: List[int] = labels_np.tolist()

    result = list(text)

    tag_shift = 1
    for prediction, terminal_idx in zip(labels, positions):
        if prediction:
            result.insert(terminal_idx + tag_shift, SENTENCE_TAG)
            tag_shift += 1

    return "".join(result)


def run_case(
    client: grpcclient.InferenceServerClient,
    text: str,
    case_name: str = "Testing!",
) -> Dict[str, float]:
    """
    Run a single test case through:
      1. Preprocessing
      2. Triton inference
      3. Postprocessing

    And report timing for each step plus total.

    Args:
        client: Triton gRPC client instance.
        text: Input text chunk.
        case_name: Human-readable name for the test case.

    Returns:
        Dictionary with timing information in milliseconds:
          {
            "case_name": str,
            "pre_ms": float,
            "infer_ms": float,
            "post_ms": float,
            "total_ms": float
          }
    """
    print(f"\n=== Running case: {case_name} ===")

    overall_start = time.time()

    # 1. Preprocessing
    pre_start = time.time()
    inputs, outputs, positions = preprocessing(text)
    pre_end = time.time()
    pre_ms = (pre_end - pre_start) * 1000.0

    # 2. Triton inference (if we have something to send)
    if inputs is None or outputs is None or not positions:
        infer_ms = 0.0
        response = None
    else:
        infer_start = time.time()
        response = client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            client_timeout=NETWORK_TIMEOUT,
            headers={},
        )
        infer_end = time.time()
        infer_ms = (infer_end - infer_start) * 1000.0

    # 3. Postprocessing
    post_start = time.time()
    output_text = postprocessing(text, positions, response)
    post_end = time.time()
    post_ms = (post_end - post_start) * 1000.0

    total_ms = (post_end - overall_start) * 1000.0

    print("--- Timing (ms) ---")
    print(f"Preprocessing:  {pre_ms:.2f} ms")
    print(f"Inference:      {infer_ms:.2f} ms")
    print(f"Postprocessing: {post_ms:.2f} ms")
    print(f"TOTAL:          {total_ms:.2f} ms")

    print("--- Segmentation results ---")
    print(f"Input:\n{text}\n")
    print(f"Output:\n{output_text}\n")

    return {
        "case_name": case_name,
        "pre_ms": pre_ms,
        "infer_ms": infer_ms,
        "post_ms": post_ms,
        "total_ms": total_ms,
    }


def warmup_model(
    client: grpcclient.InferenceServerClient,
    num_requests: int = 5,
) -> None:
    """
    Warm up the Triton model to avoid cold-start effects.

    Sends a few dummy requests with zero features.
    """
    print(f"\nWarming up model '{MODEL_NAME}' with {num_requests} dummy requests...")

    features = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    inp = grpcclient.InferInput(INPUT_NAME, [1, FEATURE_DIM], "FP32")
    inp.set_data_from_numpy(features)
    out = grpcclient.InferRequestedOutput(OUTPUT_NAME)

    for _ in range(num_requests):
        client.infer(
            model_name=MODEL_NAME,
            inputs=[inp],
            outputs=[out],
            client_timeout=NETWORK_TIMEOUT,
            headers={},
        )

    print("Warmup complete.\n")


def test_inference() -> float:
    """
    Run a suite of test cases against the Triton FIL model,
    using local preprocessing and postprocessing.

    Reports timing per case and aggregate statistics.

    Returns:
        Average total time in milliseconds across all test cases.
    """
    # Initialize gRPC client (no connection_timeout kwarg here)
    client = grpcclient.InferenceServerClient(
        url=TRITON_URL,
        verbose=False,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None,
    )

    # Workaround for older tritonclient where _stream isn't initialized
    if not hasattr(client, "_stream"):
        client._stream = None  # type: ignore[attr-defined]

    try:
        # Optional sanity checks
        if not client.is_server_live():
            raise RuntimeError("Triton server is not live")
        if not client.is_model_ready(MODEL_NAME):
            raise RuntimeError(f"Model '{MODEL_NAME}' is not ready on Triton")

        print("Connected to Triton via gRPC. Model is ready.")
        print(f"Using gRPC endpoint: {TRITON_URL}")

        # Warm up model before real measurements
        warmup_model(client, num_requests=5)

        all_metrics: List[Dict[str, float]] = []

        suite_start = time.time()

        # =======================
        # Define all test cases
        # =======================

        text_legal = (
            "Azərbaycan Respublikası Konstitusiyasının 32-ci maddəsinə əsasən, "
            "hər kəsin şəxsi və ailə həyatına hörmət hüququ vardır. "
            "Heç kəs şəxsi məlumatlarının qanunsuz toplanmasına və yayılmasına məruz qala bilməz. "
            "Məhkəmə qərarı olmadan şəxsin telefon danışıqlarına nəzarət edilməsi qadağandır."
        )
        all_metrics.append(run_case(client, text_legal, "Legal domain"))

        text_general = (
            "Bu gün hava çox gözəldir. Səhər tezdən külək əsirdi, amma indi sakitdir. "
            "Axşam yağış yağacağı proqnozlaşdırılır."
        )
        all_metrics.append(run_case(client, text_general, "General domain"))

        text_long = (
            "Azərbaycan iqtisadiyyatı son illərdə sürətli inkişaf edir. "
            "Bu inkişaf müxtəlif sahələrdə özünü göstərir. "
            "Xüsusilə texnologiya, təhsil və enerji sektorunda ciddi dəyişikliklər var. "
            "Bir çox startaplar yaranır, dövlət innovasiyalara investisiya edir. "
            "Bu proses ölkənin rəqəmsal transformasiyasını daha da gücləndirir."
        )
        all_metrics.append(run_case(client, text_long, "Long text (stress test)"))

        text_short = "Salam dünya."
        all_metrics.append(run_case(client, text_short, "Short text"))

        text_punct = "Bu nədir?! Siz bunu gördünüzmü?! Yox, inanmıram..."
        all_metrics.append(run_case(client, text_punct, "Heavy punctuation"))

        text_legal_phrases = (
            "a) Müqavilənin ləğvi; b) tərəflərin razılığı; c) məhkəmə qərarı; "
            "ç) müflislik elan edilməsi; d) qanunvericiliyin dəyişməsi; "
            "e) əhəmiyyətli şərtlərin pozulması; ə) gecikmiş öhdəlik; "
            "f) tərəflərin məsuliyyəti."
        )
        all_metrics.append(
            run_case(client, text_legal_phrases, "Legal domain — list of phrases")
        )

        text_legal_list_sentences = (
            "1) Müqavilə yalnız yazılı formada bağlandıqda etibarlı sayılır. "
            "2) Tərəflər müqavilə üzrə öhdəliklərini vicdanla yerinə yetirməlidirlər. "
            "3) Mübahisələr danışıqlar yolu ilə həll edilmədikdə, məhkəməyə müraciət oluna bilər. "
            "4) Tərəflər arasında yaranan ziyan, qanunvericiliyə uyğun olaraq kompensasiya edilir. "
            "5) Müqavilənin müddəti bitdikdə, tərəflərin yazılı razılığı ilə uzadıla bilər."
        )
        all_metrics.append(
            run_case(
                client,
                text_legal_list_sentences,
                "Legal domain — numbered list of sentences",
            )
        )

        text_legal_long_sentence = (
            "İddiaçı iddia ərizəsində göstərmişdir ki, cavabdeh tərəfindən "
            "müqavilə öhdəliklərinin vaxtında yerinə yetirilməməsi nəticəsində ona "
            "maddi ziyan dəymiş, bu ziyanın məbləği isə müstəqil auditor rəyi ilə "
            "təsdiq edilmişdir və həmin məbləğin, həmçinin gecikdirməyə görə hesablanmış "
            "dəbbə pulu və məhkəmə xərclərinin cavabdehdən tutulmasını xahiş etmişdir."
        )
        all_metrics.append(
            run_case(
                client,
                text_legal_long_sentence,
                "Legal domain — single long sentence",
            )
        )

        text_legal_abbrev = (
            "AR Mülki Məcəlləsinin 422.1-ci maddəsinə əsasən, müqavilə tərəflərinin "
            "öz öhdəliklərini lazımi qaydada yerinə yetirməsi məcburidir. "
            "Eyni Məcəllənin 439.2-ci maddəsinə görə, borclu öhdəliyi yerinə yetirmədikdə, "
            "kreditor ona qarşı məhkəməyə müraciət edə bilər. Bu halda 5 mlyn. cərimə tətbiq olur."
            "Bu Qanun və digər normativ-hüquqi aktlar (o cümlədən, \"İstehlakçıların hüquqlarının müdafiəsi haqqında\" Qanun) "
            "istehlakçıların mənafeyini qorumağa yönəlib."
        )
        all_metrics.append(
            run_case(
                client,
                text_legal_abbrev,
                "Legal domain — abbreviations & article refs",
            )
        )

        text_legal_quotes = (
            "Məhkəmə qərarında qeyd edilir ki, \"tərəflər arasında bağlanmış müqavilə "
            "bozucu şərt həyata keçənədək qüvvədə qalır\". "
            "Hakim belə nəticəyə gəlmişdir ki, cavabdehin hərəkətləri "
            "(öhdəliyin qəsdən yerinə yetirilməməsi və qarşı tərəfin ziyana salınması) "
            "qanunvericiliyin tələblərinə ziddir — bu halda, əlavə məsuliyyət tədbirləri tətbiq oluna bilər."
        )
        all_metrics.append(
            run_case(client, text_legal_quotes, "Legal domain — quotes & parentheses")
        )

        text_legal_mixed = (
            "2019-cu il 15 mart tarixli kredit müqaviləsinə əsasən, borc məbləği "
            "50 000 (əlli min) manat müəyyən edilmiş, illik faiz dərəcəsi isə 18% olmuşdur. "
            "Müqavilənin 7.2-ci bəndinə görə, borcalan 30 (otuz) gün ərzində ödənişi "
            "etmədikdə, bank gecikdirilmiş hər günə görə əlavə 0.1% dəbbə pulu hesablayır. "
            "Tərəflər arasında bu müddəa ilə bağlı hər hansı yazılı etiraz qeydə alınmamışdır."
        )
        all_metrics.append(
            run_case(
                client,
                text_legal_mixed,
                "Legal domain — dates, numbers, percentages",
            )
        )

        text_legal_many_short = (
            "İddia rədd edilir. Apellyasiya şikayəti təmin olunmur. "
            "Qərar elan olundu. Tərəflərə izah edildi. "
            "Qərardan kassasiya qaydasında şikayət vermək hüququ saxlanılır!"
        )
        all_metrics.append(
            run_case(
                client,
                text_legal_many_short,
                "Legal domain — many short sentences",
            )
        )

        text_legal_mixed_lang = (
            "Bu müqavilə Azərbaycan Respublikası qanunvericiliyinə uyğun olaraq tənzimlənir. "
            "Hər hansı dispute tərəflər arasında negotiation yolu ilə həll edilmədikdə, "
            "mübahisə Bakı Kommersiya Məhkəməsində arbitration istisna olunmaqla baxılır. "
            "Force majeure hallarına təbii fəlakətlər, müharibə və hökumət qərarları daxildir."
        )
        all_metrics.append(
            run_case(
                client,
                text_legal_mixed_lang,
                "Legal domain — mixed language & terms",
            )
        )

        suite_end = time.time()
        suite_wall_ms = (suite_end - suite_start) * 1000.0

        # =======================
        # Aggregate statistics
        # =======================

        num_cases = len(all_metrics)
        total_pre_ms = sum(m["pre_ms"] for m in all_metrics)
        total_infer_ms = sum(m["infer_ms"] for m in all_metrics)
        total_post_ms = sum(m["post_ms"] for m in all_metrics)
        total_total_ms = sum(m["total_ms"] for m in all_metrics)

        avg_pre_ms = total_pre_ms / num_cases
        avg_infer_ms = total_infer_ms / num_cases
        avg_post_ms = total_post_ms / num_cases
        avg_total_ms = total_total_ms / num_cases

        print("\n" + "=" * 60)
        print("Per-case summary (ms):")
        for m in all_metrics:
            print(
                f"- {m['case_name']}: "
                f"pre={m['pre_ms']:.2f}, "
                f"infer={m['infer_ms']:.2f}, "
                f"post={m['post_ms']:.2f}, "
                f"total={m['total_ms']:.2f}"
            )

        print("\nAggregate timings (ms):")
        print(f"Total preprocessing time:  {total_pre_ms:.2f}")
        print(f"Total inference time:      {total_infer_ms:.2f}")
        print(f"Total postprocessing time: {total_post_ms:.2f}")
        print(f"Total (sum of cases):      {total_total_ms:.2f}")
        print(f"Suite wall-clock time:     {suite_wall_ms:.2f}")

        print("\nAverage per case (ms):")
        print(f"Avg preprocessing:  {avg_pre_ms:.2f}")
        print(f"Avg inference:      {avg_infer_ms:.2f}")
        print(f"Avg postprocessing: {avg_post_ms:.2f}")
        print(f"Avg TOTAL:          {avg_total_ms:.2f}")
        print("=" * 60)

        print("\n" + "=" * 60)
        print("Connection Type: gRPC (port 8001)")
        print("NOTE: Warmup was performed before measurements.")
        print("=" * 60)

        return avg_total_ms

    finally:
        # Ensure client is properly closed to avoid destructor issues
        client.close()


if __name__ == "__main__":
    test_inference()
