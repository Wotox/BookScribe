import fitz
import numpy as np


def render_pages(pdf_path, zoom=2, page_numbers=None):
    document = fitz.open(str(pdf_path))

    try:
        matrix = fitz.Matrix(zoom, zoom)
        selected_pages = _page_indices(document.page_count, page_numbers)

        for page_index in selected_pages:
            page = document[page_index]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8)
            image = image.reshape(pixmap.height, pixmap.width, pixmap.n)
            yield page_index + 1, image
    finally:
        document.close()


def _page_indices(page_count, page_numbers):
    if page_numbers is None:
        return range(page_count)

    indices = []
    for page_number in page_numbers:
        if page_number < 1 or page_number > page_count:
            raise ValueError(f"Page {page_number} is outside the PDF page range 1-{page_count}.")
        indices.append(page_number - 1)

    return indices
