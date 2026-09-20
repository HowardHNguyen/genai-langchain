"""Parse supported uploads in memory; never log document contents."""
import logging
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZipFile
import posixpath
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from docx import Document as WordDocument
from pypdf import PdfReader
from langchain_core.documents import Document

# pypdf warnings can contain raw input bytes. Keep these out of application logs.
_pdf_logger = logging.getLogger("pypdf")
_pdf_logger.handlers = [logging.NullHandler()]
_pdf_logger.propagate = False

SUPPORTED_EXTENSIONS = ("pdf", "txt", "docx", "epub")
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 300_000
MAX_ARCHIVE_BYTES = 40 * 1024 * 1024


class DocumentLoaderException(Exception):
    pass


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _check_archive(data):
    with ZipFile(BytesIO(data)) as archive:
        if len(archive.infolist()) > 2000 or sum(i.file_size for i in archive.infolist()) > MAX_ARCHIVE_BYTES:
            raise DocumentLoaderException("The expanded document is too large. Split it into smaller files.")


def load_document(name: str, data: bytes) -> list[Document]:
    source = PurePosixPath(name.replace("\\", "/")).name[:200]
    ext = source.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentLoaderException("Unsupported format. Convert legacy .doc files to .docx.")
    if not data or len(data) > MAX_FILE_BYTES:
        raise DocumentLoaderException("Upload a nonempty file of at most 10 MB.")
    docs = []
    text_size = 0

    def add(text, **metadata):
        nonlocal text_size
        text_size += len(text)
        if text_size > MAX_TEXT_CHARS:
            raise DocumentLoaderException("The extracted text is too large. Split the document.")
        if text.strip():
            docs.append(Document(page_content=text.strip(), metadata={"source": source, **metadata}))

    try:
        if ext == "txt":
            add(data.decode("utf-8-sig"))
        elif ext == "pdf":
            pdf = PdfReader(BytesIO(data))
            if pdf.is_encrypted:
                raise DocumentLoaderException("Password-protected PDFs are unsupported. Upload an unlocked copy.")
            if len(pdf.pages) > 300:
                raise DocumentLoaderException("PDFs must contain at most 300 pages.")
            for number, page in enumerate(pdf.pages, 1):
                add(page.extract_text() or "", page=number)
        elif ext == "docx":
            _check_archive(data)
            doc = WordDocument(BytesIO(data))
            text = []
            for block in doc.iter_inner_content():
                if hasattr(block, "text"):
                    text.append(block.text)
                else:
                    text.extend(" | ".join(cell.text for cell in row.cells) for row in block.rows)
            add("\n".join(text))
        else:
            _check_archive(data)
            with ZipFile(BytesIO(data)) as archive:
                container = ET.fromstring(archive.read("META-INF/container.xml"))
                rootfile = next(e for e in container.iter() if e.tag.endswith("}rootfile"))
                opf_path = rootfile.attrib["full-path"]
                package = ET.fromstring(archive.read(opf_path))
                manifest = {e.attrib["id"]: e.attrib for e in package.iter() if e.tag.endswith("}item")}
                for index, item in enumerate((e for e in package.iter() if e.tag.endswith("}itemref")), 1):
                    entry = manifest[item.attrib["idref"]]
                    if entry.get("media-type") != "application/xhtml+xml":
                        continue
                    path = posixpath.normpath(posixpath.join(posixpath.dirname(opf_path), entry["href"]))
                    parser = _Text()
                    parser.feed(archive.read(path).decode("utf-8"))
                    add(" ".join(parser.parts), section=index)
    except DocumentLoaderException:
        raise
    except Exception:
        raise DocumentLoaderException("Could not parse this file. Use a valid PDF, UTF-8 TXT, DOCX, or EPUB.") from None
    if not docs:
        raise DocumentLoaderException("No readable text found. Scanned PDFs need OCR before upload.")
    return docs
