def load_quran():
    quran = {}

    with open("datasets/quran-text/quran-simple.txt", "r", encoding="utf-8") as f:
        for line in f:

            parts = line.strip().split("|")

            # skip invalid lines
            if len(parts) != 3:
                continue

            surah, ayah, text = parts

            key = f"{surah}{ayah.zfill(3)}"
            quran[key] = text

    return quran