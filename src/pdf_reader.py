import json
import os
import re
import shutil
import subprocess
import sys


_IMAGE_PATTERN = re.compile(r"!\[[^]]*\]\([^)]*\)")
_PAGE_PATTERN = re.compile(r"<!--\s*page(?:\s+|:)[^>]*-->", re.IGNORECASE)


def _mineru_command():
    executable = shutil.which("mineru")
    if executable:
        return [executable]
    return [sys.executable, "-m", "mineru.cli.main"]


def _is_table_line(line):
    return "|" in line and bool(line.strip())


def _segments_from_markdown(markdown):
    segments = []
    text_lines = []
    table_lines = []

    def flush_text():
        text = "\n".join(text_lines).strip()
        if text:
            segments.append({"text": text, "type": "TEXT"})
        text_lines.clear()

    def flush_table():
        table = "\n".join(table_lines).strip()
        if table:
            segments.append({"text": table, "type": "TABLE"})
        table_lines.clear()

    for line in markdown.splitlines():
        image_match = _IMAGE_PATTERN.search(line)
        if image_match:
            flush_text()
            flush_table()
            segments.append({"text": line.strip(), "type": "IMAGE"})
        elif _is_table_line(line):
            flush_text()
            table_lines.append(line)
        else:
            flush_table()
            text_lines.append(line)

    flush_text()
    flush_table()
    return segments


def _read_pdf_with_pymupdf(pdf_path):
    """Fallback text extraction when the optional MinerU server is offline."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "MinerU is not running and PyMuPDF is not installed. "
            "Run 'mineru server start' or install PyMuPDF."
        ) from exc

    document = fitz.open(pdf_path)
    segments = []
    image_count = 0
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        image_count += len(page.get_images(full=True))
        if text:
            segments.append({
                "text": text,
                "type": "TEXT",
                "modality": "TEXT",
                "page": page_number,
            })
    stats = {
        "pages": len(document),
        "tables": 0,
        "images": image_count,
        "fallback": "PyMuPDF",
    }
    document.close()
    return segments, stats


def read_pdf(pdf_path):
    """Parse a PDF with MinerU and return the project's typed segment format."""
    mineru_command = _mineru_command()
    parse_command = mineru_command + [
        "parse", os.fspath(pdf_path), "--remote", "--tier", "standard",
        "--pages", "all", "--json",
        "--wait", os.environ.get("MINERU_WAIT_SECONDS", "300"), "--force",
    ]

    commands = [parse_command]
    last_error = None
    result = None
    cache_repaired = False
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            break
        except FileNotFoundError as exc:
            raise RuntimeError(
                "MinerU is not installed or is not available on PATH. "
                "Install MinerU 4 before running ingestion."
            ) from exc
        except subprocess.CalledProcessError as exc:
            last_error = exc
            details = (exc.stderr or exc.stdout).strip()
            if "server_not_running" in details:
                print(
                    f"MinerU server is offline for {pdf_path}; "
                    "using PyMuPDF text extraction."
                )
                return _read_pdf_with_pymupdf(pdf_path)
            if "invalid_middle_json_output" not in details:
                raise RuntimeError(f"MinerU failed to parse {pdf_path}: {details}") from exc

            if not cache_repaired:
                try:
                    subprocess.run(
                        mineru_command + [
                            "forget", os.fspath(pdf_path), "--no-dry-run", "--json"
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                    )
                except subprocess.CalledProcessError as forget_error:
                    forget_details = (
                        forget_error.stderr or forget_error.stdout
                    ).strip()
                    raise RuntimeError(
                        f"MinerU returned invalid cached data for {pdf_path}, "
                        f"and the cache could not be cleared: {forget_details}"
                    ) from forget_error
                cache_repaired = True
                commands.append(parse_command)

    if result is None and last_error is not None:
        details = (last_error.stderr or last_error.stdout).strip()
        raise RuntimeError(
            f"MinerU could not parse {pdf_path} after clearing its cached result: {details}. "
            "The configured parse server supports only the standard tier."
        ) from last_error

    try:
        payload = json.loads(result.stdout)
        markdown = payload["content"]["content"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("MinerU returned an unexpected JSON response.") from exc

    segments = _segments_from_markdown(markdown)
    stats = {
        "pages": len(_PAGE_PATTERN.findall(markdown)),
        "tables": sum(segment["type"] == "TABLE" for segment in segments),
        "images": sum(segment["type"] == "IMAGE" for segment in segments),
    }
    return segments, stats