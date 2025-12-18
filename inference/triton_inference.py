import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import tritonclient.grpc as grpcclient

# Add parent directory to path to import azcharboundary
sys.path.insert(0, str(Path(__file__).parent.parent))

from azcharboundary.utils.constants import TERMINAL_SENTENCE_CHAR_LIST, SENTENCE_TAG
from azcharboundary.utils.features import FeatureExtractor


# Triton gRPC endpoint and model name for the FIL model
TRITON_URL: str = "localhost:8001"
MODEL_NAME: str = "model"

# Connection configuration
NETWORK_TIMEOUT = 60.0  # seconds

# Model I/O configuration (do NOT change names!)
INPUT_NAME = "input__0"
OUTPUT_NAME = "output__0"
FEATURE_DIM = 28

# Single global feature extractor instance reused across calls
feature_extractor = FeatureExtractor()


def preprocessing(
    text: str,
    left_window: int = 9,
    right_window: int = 9,
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
    print(features_np)

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

    print(positions)
    print(probs)
    print(labels)

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
    Run the same 5 sequential test cases as in the local TextSegmenter
    inference.py against the Triton FIL model, using local preprocessing
    and postprocessing.

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

        # CASE 1 — Legal contract, dates, abbreviations, mixed list styles
        text_case_1 = (
            "Bu Müqavilə 12.05.2015-ci il tarixdə Bakı şəh., AZ0000, Nizami k., 15 ünvanında"
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
        all_metrics.append(
            run_case(
                client,
                text_case_1,
                "Case 1 — Contract + lists + dates",
            )
        )

        # # CASE 2 — Court decision with numbered paragraphs, internal abbreviations and mixed punctuation
        # text_case_2 = (
        #     "Bakı Apellyasiya Məhk. məhkəmə heyəti, hakim İ.Xəlilovun sədrliyi ilə açıq məhkəmə iclasında "
        #     "iddiaçı A.A.-nın cavabdeh “Gamma” MMC-yə qarşı iddiası üzrə işi baxaraq müəyyən etdi ki, "
        #     "mübahisə predmeti 15.09.2019-cu il tarixli satınalma müqaviləsinin icrası ilə bağlıdır. "
        #     "1) İddiaçı bildirmişdir ki, cavabdeh öhdəliyi vaxtında yerinə yetirməyib, nəticədə 5 000 (beş min) manat "
        #     "maddi ziyan dəymişdir; 2) Cavabdeh isə, öz növbəsində, öhdəliyin pozulmasını force majeure ilə əsaslandırmış, "
        #     "lakin bu barədə hər hansı rəsmi sübut (sertifikat, akt və s.) təqdim etməmişdir. "
        #     "Məhkəmə hesab edir ki, cavabdehin arqumentləri əsassızdır... "
        #     "Nəticə etibarilə, iddia qismən təmin edilir?! "
        #     "Qərar elan olundu və tərəflərə izah edildi ki, qərardan 1 (bir) ay müddətində 6m. Əməl. kassasiya şikayəti verilə bilər."
        # )
        # all_metrics.append(
        #     run_case(
        #         client,
        #         text_case_2,
        #         "Case 2 — Court decision, numbered items",
        #     )
        # )

        # # CASE 3 — Mixed language, percentages, times, inline list items, tricky abbreviations
        # text_case_3 = (
        #     "12.03.2020 tarixli kredit müqaviləsinə (№ KM-2020/03-12) əsasən, borc məbləği 75 000 (yetiş beş min) manat "
        #     "təyin edilmişdir. İllik faiz dərəcəsi 18,5% olaraq müəyyən edilib; gecikmə halında isə əlavə 0,1% dəbbə pulu "
        #     "hesablanır. "
        #     "Clause 5.2-də qeyd olunur: \"Borc veren shall provide audited financial statements\" — lakin azərbaycanca "
        #     "versiyada “audit edilmiş maliyyə hesabatı” ifadəsi istifadə olunmuşdur. "
        #     "Saat 10:30-da tərəflər bankın mərkəzi ofisində (Bakı ş., Heydər Əliyev pr., 10) görüşərək aşağıdakıları "
        #     "razılaşdırmışlar: (i) ödəniş qrafiki yenidən tərtib olunur; (ii) 3 (üç) ay müddətinə grace period tətbiq edilir; "
        #     "(iii) borcalanın əlavə təminat təqdim etməsi tələb edilmir. "
        #     "Bu protokol, Bank Nəzarəti Şöb., həmçinin Risk Dept. tərəfindən də təsdiq edilib."
        # )
        # all_metrics.append(
        #     run_case(
        #         client,
        #         text_case_3,
        #         "Case 3 — Mixed language, %, times, lists",
        #     )
        # )

        # # CASE 4 — Long narrative, quotes, parentheses, ellipses, heavy punctuation, fake sentence-like abbreviations
        # text_case_4 = (
        #     "Məhkəmə iclasında cavabdeh belə demişdir: \"Mən müqaviləni oxumuşam, lakin oradakı ‘7.2-ci bənd’ "
        #     "mənə aydın olmayıb\". Hakim sual verir: \"Siz hüquqşünasla məsləhətləşmisinizmi?!\" "
        #     "Cavabdeh cavab verir ki, o, yalnız tanışı olan bir mütəxəssislə (prof. S.Əliyev) qısa müzakirə aparıb, "
        #     "lakin rəsmi legal opinion almamışdır. "
        #     "Zalda olan nümayəndə (şirk. nümay., yəni rəsmi təmsilçi) isə bildirir ki, tərəflər arasında "
        #     "‘gentlemen’s agreement’ də olub... Lakin bu, yazılı formada təsdiq edilməyib. "
        #     "Hakim qeyd edir ki, belə informal razılaşmalar AR qanunvericiliyində ayrıca təsbit olunmayıb, "
        #     "bu səbəbdən də məhk. onları hüquqi əsas kimi qəbul etmir. "
        #     "Bu halda, yalnız müqavilənin mətni, əlavə razılaşmalar (Annex 1, Annex 2 və s.) və tərəflərin faktiki davranışı "
        #     "nəzərə alınır."
        # )
        # all_metrics.append(
        #     run_case(
        #         client,
        #         text_case_4,
        #         "Case 4 — Quotes, ellipses, fake endings",
        #     )
        # )

        # # CASE 5 — Mixed obligations, bullets, broken structures, dates and no-punct fragments
        # text_case_5 = (
        #     "“Delta Logistic” MMC ilə bağlanmış 05.11.2018-ci il tarixli xidmət müqaviləsinin 3-cü bölməsi "
        #     "öhdəliklərin icrasına həsr olunmuşdur. Bölmə aşağıdakı bəndlərdən ibarətdir: "
        #     "1) Podratçı yükün təhlükəsiz daşınmasını təmin etməlidir — yük itdikdə və ya zədələndikdə, "
        #     "Mülki Məcəllənin 921-ci maddəsinə uyğun olaraq məsuliyyət daşıyır; "
        #     "2) Müştəri xidmət haqqını 30 (otuz) təqvim günü ərzində ödəməlidir; "
        #     "3) tərəflər aşağıdakı hallarda müqaviləni birtərəfli qaydada ləğv edə bilərlər: "
        #     "a) müflislik elan edilməsi; b) 60 gündən artıq gecikmə; c) qanunvericiliyin dəyişməsi nəticəsində "
        #     "müqavilənin icrasının faktiki olaraq mümkünsüz olması. "
        #     "Bundan əlavə, qeydlər bölməsində belə yazılmışdır: "
        #     "“Əlavə xidmətlər göstərilə bilər qiymət sonradan razılaşdırılır hər bir tərəf bu barədə əvvəlcədən yazılı "
        #     "məlumat təqdim etməlidir” — cümlə ardıcıllığı pozulmuş, nöqtə və vergüllər isə, praktiki olaraq, "
        #     "heç yerdə qoyulmamışdır. "
        #     "Sonda 01.01.2019 tarixli əlavə razılaşma ilə qiymətlər 10% artırılmış, lakin əvvəlki qrafik dəyişdirilməmişdir."
        # )
        # all_metrics.append(
        #     run_case(
        #         client,
        #         text_case_5,
        #         "Case 5 — Bullets, broken text, dates",
        #     )
        # )

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