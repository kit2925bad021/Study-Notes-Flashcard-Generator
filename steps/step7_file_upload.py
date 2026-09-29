"""
Step 7 — File Upload & Text Extraction
------------------------------------------
A reusable module for uploading a study file (TXT, PDF, or DOCX) and
extracting its raw text so it can be used elsewhere in the app later.

This step does NOT call any AI model and does NOT generate any study
material, flashcards, or quizzes. It only uploads a file, pulls the
text out of it, and cleans that text into normal readable sentences.
The uploaded material is NEVER turned into code: what you see in the
preview is the real text of the file, nothing invented or rewritten.

Requires:  pip install pypdf python-docx

Imported in app.py with:
    from steps.step7_file_upload import show_file_upload
"""

import io
import re

import streamlit as st
import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

# How many characters of extracted text to show in the preview,
# so the page doesn't get huge for long documents.
PREVIEW_CHARACTER_LIMIT = 5000

# Shown whenever a file contains no usable text.
NO_TEXT_MESSAGE = "No readable text was found in this file."

# If ANY other part of the app sends the extracted study text to Ollama,
# put this instruction in the prompt so the model answers in plain
# natural language and never in code. (See build_plain_language_prompt.)
PLAIN_LANGUAGE_INSTRUCTION = (
    "Generate educational study content in plain natural language. "
    "Do not output Python, JavaScript, HTML, JSON, SQL, or programming code."
)

# Characters that PDFs / Word files often contain and that look like
# garbage when shown as plain text.
_CHARACTER_FIXES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u00a0": " ",   # non-breaking space
    "\u2007": " ",
    "\u202f": " ",
    "\u00ad": "",    # soft hyphen
    "\u200b": "",    # zero-width space
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",    # byte-order mark
    "\ufffd": "",    # replacement character (failed decoding)
    "\u2028": "\n",  # unicode line / paragraph separators
    "\u2029": "\n",
    "\f": "\n",      # form feed (page break)
    # Bullet symbols that Word / PDFs store as private-use "Symbol font" characters.
    "\uf0b7": "•",
    "\uf0a7": "•",
    "\uf0d8": "•",
    "\uf076": "•",
    "\uf0fc": "•",
}

# A line that starts a bullet point or a numbered/lettered list item.
_LIST_ITEM = re.compile(r"^(?:[•●◦▪■□‣∙·*\-–—]|\(?\d{1,3}[.)]|\(?[A-Za-z][.)])\s+")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _normalise_characters(text):
    """Fix line endings, ligatures, odd spaces, invisible and control characters."""
    # Line endings FIRST. Windows files use "\r\n"; turning "\r" into "\n"
    # one character at a time would double every line break.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    for bad, good in _CHARACTER_FIXES.items():
        text = text.replace(bad, good)

    # Unreadable font-glyph placeholders that some PDFs produce, e.g. "(cid:123)".
    text = re.sub(r"\(cid:\d+\)", "", text)
    # Any remaining private-use characters are unreadable symbols, not words.
    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    # Remove remaining control characters (keep newline and tab).
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def _tidy_lines(text):
    """
    Collapse repeated spaces and remove leading indentation.

    Removing the indentation matters: four leading spaces would be shown
    as a grey code block by Streamlit's markdown renderer.
    """
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    return "\n".join(lines)


def _looks_like_heading(line):
    """True for a short, capitalised line that does not end like a sentence."""
    return (
        len(line) <= 70
        and len(line.split()) <= 9
        and (line[0].isupper() or line[0].isdigit())
        and not re.search(r"[.!?,;]$", line)
    )


def _reflow_pdf_paragraphs(text):
    """
    PDFs break every visual line with a newline, even in the middle of a
    sentence. Join those broken lines back into real paragraphs, while
    keeping headings, short lines and list items on their own lines.
    """
    lengths = sorted(len(line) for line in text.split("\n") if line)
    if len(lengths) < 5:
        return text

    typical_width = lengths[int(len(lengths) * 0.9)]
    if typical_width < 30:
        return text  # very short lines: probably a list or poem, leave it

    output_blocks = []

    for block in re.split(r"\n{2,}", text):
        # Each item is [text, is_list_item]
        items = []
        for line in block.split("\n"):
            if not line:
                continue
            previous = items[-1][0] if items else None

            if previous is None:
                items.append([line, bool(_LIST_ITEM.match(line))])
            elif _LIST_ITEM.match(line):
                items.append([line, True])
            elif line[0].islower():
                # Starts in lower case: it continues the previous line.
                if previous.endswith("-") and len(previous) > 1 and previous[-2].isalpha():
                    items[-1][0] = previous[:-1] + line  # join "exam-" + "ple"
                else:
                    items[-1][0] = previous + " " + line
            elif _looks_like_heading(line) and re.search(r"[.!?:\"')\u201d]$", previous):
                # A short title-like line after a finished sentence: new heading.
                items.append([line, False])
            elif len(previous) >= 0.6 * typical_width and not items[-1][1]:
                # Previous line was nearly full width: same paragraph.
                items[-1][0] = previous + " " + line
            elif items[-1][1] and len(previous) >= 0.6 * typical_width:
                # A wrapped list item.
                items[-1][0] = previous + " " + line
            else:
                items.append([line, False])

        # Consecutive list items stay together; everything else is a paragraph.
        pieces = []
        for text_part, is_item in items:
            if is_item and pieces and pieces[-1][1]:
                pieces[-1][0] += "\n" + text_part
            else:
                pieces.append([text_part, is_item])
        output_blocks.extend(piece[0] for piece in pieces)

    return "\n\n".join(output_blocks)


def clean_extracted_text(text, reflow=False):
    """
    Turn raw extracted text into clean, readable study text.

    - fixes line endings, ligatures, odd spaces and control characters
    - removes extra spaces and leading indentation
    - removes unnecessary blank lines (keeps ONE blank line between paragraphs)
    - keeps headings, paragraphs, terminology and meaning unchanged
    - reflow=True (used for PDFs) joins lines that were broken mid-sentence

    No words are added, removed or rewritten.
    """
    if not text:
        return ""

    text = _normalise_characters(text)
    text = _tidy_lines(text)
    if reflow:
        text = _reflow_pdf_paragraphs(text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _has_readable_text(text):
    """True if the text contains at least one real word (2+ letters in a row)."""
    return bool(text) and re.search(r"[^\W\d_]{2,}", text) is not None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_text_from_txt(uploaded_file):
    """
    Read a plain text (.txt) file and return its contents as a string.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        str: The text contents of the file.
    """
    # getvalue() always returns the whole file, even if the file object
    # was already read once during an earlier Streamlit rerun.
    raw_bytes = uploaded_file.getvalue()

    # UTF-16 is only used when the file really starts with a UTF-16 byte-order
    # mark. (Trying it blindly can "succeed" on ordinary files and produce
    # unreadable garbage.)
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw_bytes.decode("utf-16")
        except UnicodeError:
            pass

    # utf-8-sig handles UTF-8 with or without a BOM; cp1252 covers most
    # Windows text files; latin-1 can decode any byte, so it never fails.
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeError:
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


def _drop_page_number_lines(page_text):
    """Remove a first/last line that is only a page number (e.g. '42')."""
    lines = page_text.strip().split("\n")
    if len(lines) > 1 and re.fullmatch(r"\s*\d{1,4}\s*", lines[-1]):
        lines = lines[:-1]
    if len(lines) > 1 and re.fullmatch(r"\s*\d{1,4}\s*", lines[0]):
        lines = lines[1:]
    return "\n".join(lines)


def extract_text_from_pdf(uploaded_file):
    """
    Extract the real text from every page of a PDF file using pypdf.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        str: The combined text from all pages of the PDF.

    Raises:
        ValueError: If the PDF is password protected.
    """
    pdf_bytes = io.BytesIO(uploaded_file.getvalue())
    pdf_reader = PdfReader(pdf_bytes)

    if pdf_reader.is_encrypted:
        # Many "protected" PDFs open with an empty password.
        try:
            unlocked = int(pdf_reader.decrypt("")) != 0
        except Exception:
            unlocked = False
        if not unlocked:
            raise ValueError("This PDF is password protected. Please upload an unlocked copy.")

    all_text = []
    for page in pdf_reader.pages:
        try:
            page_text = page.extract_text()
        except Exception:
            page_text = None  # skip a damaged page instead of failing everything
        if page_text and page_text.strip():
            all_text.append(_drop_page_number_lines(_normalise_characters(page_text)))

    return "\n\n".join(all_text)


def _iter_docx_blocks(document):
    """Yield the paragraphs and tables of a Word file in reading order."""
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield Table(child, document)


def _is_docx_list_paragraph(paragraph):
    """True for a Word bullet / numbered-list paragraph (never for headings)."""
    try:
        style_name = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
    except Exception:
        style_name = ""

    if style_name.startswith(("heading", "title", "subtitle")):
        return False
    if style_name.startswith("list"):
        return True

    properties = paragraph._p.pPr
    return properties is not None and properties.numPr is not None


def extract_text_from_docx(uploaded_file):
    """
    Extract text from every paragraph (and table) of a Word (.docx) file
    using python-docx.

    Paragraphs and headings are separated by one blank line; bullet and
    numbered items stay together, one per line.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        str: The combined text from the document, in reading order.
    """
    docx_bytes = io.BytesIO(uploaded_file.getvalue())
    document = docx.Document(docx_bytes)

    # Each entry is (text, is_list_item).
    blocks = []
    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            paragraph_text = block.text.strip()
            if not paragraph_text:
                continue
            is_item = _is_docx_list_paragraph(block)
            if is_item and not _LIST_ITEM.match(paragraph_text):
                paragraph_text = "- " + paragraph_text
            blocks.append((paragraph_text, is_item))
        else:
            rows = []
            for row in block.rows:
                cells = []
                seen = set()
                for cell in row.cells:
                    if cell._tc in seen:  # merged cells repeat
                        continue
                    seen.add(cell._tc)
                    cell_text = cell.text.strip().replace("\n", " ")
                    if cell_text:
                        cells.append(cell_text)
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append(("\n".join(rows), False))

    pieces = []
    for index, (block_text, is_item) in enumerate(blocks):
        if index > 0:
            # List items sit on consecutive lines; everything else gets a blank line.
            pieces.append("\n" if (is_item and blocks[index - 1][1]) else "\n\n")
        pieces.append(block_text)
    return "".join(pieces)


def extract_text_from_file(uploaded_file):
    """
    Detect the uploaded file's type from its filename, extract the text
    with the correct function, and return it cleaned.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        str: Clean, readable text. An empty string means the file has no
             readable text (for example a scanned PDF).

    Raises:
        ValueError: If the file type is not supported.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        text = clean_extracted_text(extract_text_from_txt(uploaded_file))
    elif filename.endswith(".pdf"):
        text = clean_extracted_text(extract_text_from_pdf(uploaded_file), reflow=True)
    elif filename.endswith(".docx"):
        text = clean_extracted_text(extract_text_from_docx(uploaded_file))
    else:
        raise ValueError(
            "Unsupported file type. Please upload a .txt, .pdf, or .docx file."
        )

    # Nothing but symbols / unreadable glyphs counts as "no text".
    return text if _has_readable_text(text) else ""


# ---------------------------------------------------------------------------
# Helpers for other parts of the app (optional to use)
# ---------------------------------------------------------------------------

def get_uploaded_study_text():
    """Return the text extracted on this page ('' if nothing was uploaded)."""
    return st.session_state.get("uploaded_text") or ""


def build_plain_language_prompt(task, study_text):
    """
    Build an Ollama prompt that asks for plain study content, never code.

    Use it only where plain readable study text is wanted (notes, summaries,
    explanations). Do not add it to prompts that must return a structured
    format such as JSON.
    """
    return (
        f"{PLAIN_LANGUAGE_INSTRUCTION}\n\n"
        f"{task.strip()}\n\n"
        f"STUDY MATERIAL:\n{study_text}"
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _preview_as_markdown(text):
    """
    Make plain text safe to show with st.write().

    st.write() treats text as markdown, so characters like `, #, *, _ and
    indented lines would turn into code blocks, headings or bold text.
    Escaping them makes the preview show the words exactly as extracted.
    """
    escaped = re.sub(r"([\\`*_~$<>|#\[\]])", r"\\\1", text)
    # Keep single line breaks (needed for list items) inside a paragraph.
    return escaped.replace("\n", "  \n")


def _forget_saved_upload():
    """Clear the saved text so no other page can use an old file by mistake."""
    st.session_state.uploaded_text = None
    st.session_state.uploaded_filename = None
    st.session_state.uploaded_signature = None


def show_file_upload():
    """Display the file upload interface and extract text from the uploaded file."""

    # ---- Session state for this page ----
    # Holds the extracted text and the original filename so other
    # parts of the app can use them later.
    if "uploaded_text" not in st.session_state:
        st.session_state.uploaded_text = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None
    if "uploaded_signature" not in st.session_state:
        st.session_state.uploaded_signature = None

    uploaded_file = st.file_uploader(
        "📤 Upload Study Material",
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file is not None:
        st.write(f"**File:** {uploaded_file.name}")

        # Try to extract the text; show a friendly error if it fails
        try:
            # Only read the file again if it is a new upload (Streamlit
            # reruns this function on every click).
            signature = (uploaded_file.name, uploaded_file.size)
            if (
                st.session_state.uploaded_signature == signature
                and st.session_state.uploaded_text
            ):
                extracted_text = st.session_state.uploaded_text
            else:
                extracted_text = extract_text_from_file(uploaded_file)

            if not extracted_text.strip():
                # Never keep the previous file's text when this file has none.
                _forget_saved_upload()
                st.warning(f"⚠️ {NO_TEXT_MESSAGE}")
                st.caption(
                    "If this is a scanned PDF (pictures of pages), it needs OCR first."
                )
            else:
                # Save the extracted text and filename for later use
                st.session_state.uploaded_text = extracted_text
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.uploaded_signature = signature

                st.success(f"✅ Extracted {len(extracted_text)} characters from this file.")

                # Show the extracted text BEFORE it is used by any other feature.
                with st.expander("👁️ Preview Extracted Text", expanded=True):
                    preview_text = extracted_text[:PREVIEW_CHARACTER_LIMIT]
                    st.write(_preview_as_markdown(preview_text))
                    if len(extracted_text) > PREVIEW_CHARACTER_LIMIT:
                        st.caption(
                            f"Showing the first {PREVIEW_CHARACTER_LIMIT} characters "
                            f"of {len(extracted_text)} total."
                        )

        except ValueError as e:
            # Unsupported file type or password-protected PDF
            _forget_saved_upload()
            st.error(f"⚠️ {e}")
        except Exception:
            # Any other extraction problem (corrupted file, etc.)
            _forget_saved_upload()
            st.error("⚠️ Something went wrong while reading this file. Please try a different file.")