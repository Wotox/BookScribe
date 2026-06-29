import easyocr


class OCRReader:
    def __init__(self, languages=None, gpu=True):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None

    def get_reader(self):
        if self._reader is None:
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)

        return self._reader

    def read_text_from_page(self, image):
        results = self.get_reader().readtext(image, detail=0, paragraph=True)
        lines = [text.strip() for text in results if text.strip()]
        return "\n\n".join(lines)
