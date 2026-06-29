import fitz
import numpy as np


def render_pages(pdf_path, zoom=2):
    document = fitz.open(str(pdf_path))

    try:
        matrix = fitz.Matrix(zoom, zoom)

        for page_index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8)
            image = image.reshape(pixmap.height, pixmap.width, pixmap.n)
            yield page_index + 1, image
    finally:
        document.close()
