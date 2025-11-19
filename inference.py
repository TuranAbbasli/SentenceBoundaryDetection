from azcharboundary.segmenter import TextSegmenter

def test_inference_legal_domain(segmenter_dir: str):
    # Hüquqi domenə aid nümunə mətn
    text = (
        "Azərbaycan Respublikası Konstitusiyasının 32-ci maddəsinə əsasən, "
        "hər kəsin şəxsi və ailə həyatına hörmət hüququ vardır. "
        "Heç kəs şəxsi məlumatlarının qanunsuz toplanmasına və yayılmasına "
        "məruz qala bilməz. "
        "Məhkəmə qərarı olmadan şəxsin telefon danışıqlarına nəzarət edilməsi qadağandır."
    )

    # Segmenteri yaradın
    segmenter = TextSegmenter()

    segmenter.load(segmenter_dir)

    # inference icra olunur
    output = segmenter.inference(text=text, threshold=0.5)

    print("\n--- Segmentation results ---")
    print(output)

if __name__ == "__main__":
    model_path = r"azcharboundary\models\model.xz"
    test_inference_legal_domain(segmenter_dir=model_path)