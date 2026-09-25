import openpyxl
import os
import re


def read_excel(excel_path):

    wb = openpyxl.load_workbook(
        excel_path,
        data_only=True
    )

    segments = []
    sheet_count = 0

    for sheet_name in wb.sheetnames:

        sheet = wb[sheet_name]
        sheet_count += 1

        table_text = (
            f"Workbook: {os.path.basename(excel_path)}\n"
            f"Sheet: {sheet_name}\n"
        )

        rows = list(
            sheet.iter_rows(values_only=True)
        )

        if not rows:
            continue

        header_row_index = 0
        for row_index in range(len(rows) - 1):
            current_values = [value for value in rows[row_index] if value is not None]
            next_values = [value for value in rows[row_index + 1] if value is not None]
            if len(current_values) >= 2 and len(next_values) >= 2:
                header_row_index = row_index
                break

        raw_headers = list(rows[header_row_index])
        first_column = next(
            (index for index, value in enumerate(raw_headers) if value is not None),
            0,
        )
        headers = [
            str(header).strip() if header else f"Column {column_index}"
            for column_index, header in enumerate(raw_headers[first_column:], start=first_column + 1)
        ]

        for preamble_index, preamble_row in enumerate(rows[:header_row_index], start=1):
            values = [str(value) for value in preamble_row if value is not None]
            if values:
                table_text += f"Preamble row {preamble_index}: " + " | ".join(values) + "\n"

        table_text += "Columns: " + " | ".join(headers) + "\n"
        table_text += "=" * 80 + "\n"

        last_project = ""

        for row_index, row in enumerate(rows[header_row_index + 1:], start=header_row_index + 2):

            row_values = list(row[first_column:])

            data = dict(zip(headers, row_values))

            project = str(
                data.get("Project Name", "")
            ).strip()

            if not project:
                project = last_project
            else:
                last_project = project

            fixed_row = []

            for header, value in zip(headers, row_values):

                if header == "Project Name":
                    fixed_row.append(f"{header} = {project}")
                else:
                    display_value = "" if value is None else str(value)
                    fixed_row.append(f"{header} = {display_value}")

            table_text += (
                f"Row {row_index}: "
                + " | ".join(fixed_row)
                + "\n"
            )

        segments.append({
            "text": table_text,
            "type": "TABLE",
            "modality": "TABLE",
            "sheet": sheet_name
        })

    wb.close()

    stats = {
        "pages": 0,
        "tables": sheet_count,
        "sheets": len(segments),
        "images": 0
    }

    return segments, stats


def _normalize_value(value):

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower()
    )


def _find_header_index(rows):
    for row_index in range(len(rows) - 1):
        current_values = [value for value in rows[row_index] if value is not None]
        next_values = [value for value in rows[row_index + 1] if value is not None]
        if len(current_values) >= 2 and len(next_values) >= 2:
            return row_index
    return 0


def _headers_from_row(row):
    first_column = next(
        (index for index, value in enumerate(row) if value is not None),
        0,
    )
    headers = [
        str(header).strip() if header else f"Column {column_index}"
        for column_index, header in enumerate(row[first_column:], start=first_column + 1)
    ]
    return first_column, headers


def compare_workbooks(workbook_paths):
    """Return a readable comparison of the supplied Excel workbooks."""
    valid_paths = [
        path for path in workbook_paths
        if not os.path.basename(path).startswith("~$")
    ]
    triples = workbook_relationship_triples(valid_paths)

    lines = ["EXCEL WORKBOOK COMPARISON:"]
    if not valid_paths:
        return "\n".join(lines + ["No Excel workbooks found."])

    lines.append("Workbooks: " + ", ".join(
        os.path.basename(path) for path in valid_paths
    ))
    if not triples:
        lines.append("No shared columns, values, or worksheets were found.")
        return "\n".join(lines)

    for triple in triples:
        detail = f" ({triple['details']})" if triple.get("details") else ""
        lines.append(
            f"- {triple['subject']} {triple['relation']} "
            f"{triple['object']}{detail}"
        )
    return "\n".join(lines)


def workbook_relationship_triples(workbook_paths):

    workbook_data = []

    for path in workbook_paths:

        if os.path.basename(path).startswith("~$"):
            continue

        workbook = openpyxl.load_workbook(
            path,
            data_only=True,
            read_only=True
        )

        headers = set()
        values = set()
        sheets = set()

        for sheet in workbook.worksheets:

            rows = [
                [cell.value for cell in row]
                for row in sheet.iter_rows()
                if any(cell.value is not None for cell in row)
            ]

            if not rows:
                continue

            sheets.add(sheet.title)

            header_index = _find_header_index(rows)
            first_column, header_names = _headers_from_row(rows[header_index])

            headers.update(
                _normalize_value(v)
                for v in header_names
                if _normalize_value(v)
            )

            values.update(
                _normalize_value(v)
                for row in rows[header_index + 1:]
                for v in row[first_column:]
                if _normalize_value(v)
            )

        workbook.close()

        workbook_data.append({
            "name": os.path.basename(path),
            "headers": headers,
            "values": values,
            "sheets": sheets
        })

    triples = []

    for i, left in enumerate(workbook_data):

        for right in workbook_data[i + 1:]:

            common_headers = sorted(
                left["headers"].intersection(
                    right["headers"]
                )
            )

            shared_values = sorted(
                left["values"].intersection(
                    right["values"]
                )
            )

            shared_sheets = sorted(
                left["sheets"].intersection(
                    right["sheets"]
                )
            )

            if common_headers:

                triples.append({
                    "subject": left["name"],
                    "relation": "shares_columns_with",
                    "object": right["name"],
                    "source": left["name"],
                    "details": ", ".join(common_headers[:20]),
                    "modality": "TABLE"
                })

            if shared_values:

                triples.append({
                    "subject": left["name"],
                    "relation": "shares_values_with",
                    "object": right["name"],
                    "source": left["name"],
                    "details": ", ".join(shared_values[:20]),
                    "modality": "TABLE"
                })

            if shared_sheets:

                triples.append({
                    "subject": left["name"],
                    "relation": "shares_worksheets_with",
                    "object": right["name"],
                    "source": left["name"],
                    "details": ", ".join(shared_sheets),
                    "modality": "TABLE"
                })

    return triples


def excel_entity_triples(workbook_paths):

    triples = []

    for path in workbook_paths:

        workbook = openpyxl.load_workbook(
            path,
            data_only=True
        )

        workbook_name = os.path.basename(path)

        for sheet in workbook.worksheets:

            rows = list(
                sheet.iter_rows(values_only=True)
            )

            if len(rows) < 2:
                continue

            header_index = _find_header_index(rows)
            first_column, headers = _headers_from_row(rows[header_index])

            last_project = ""

            for row in rows[header_index + 1:]:

                data = dict(zip(headers, row[first_column:]))

                project = str(
                    data.get("Project Name", "")
                ).strip()

                if not project:
                    project = last_project
                else:
                    last_project = project

                task = str(
                    data.get("Task Name", "")
                ).strip()

                person = str(
                    data.get("Assigned to", "")
                ).strip()

                progress = str(
                    data.get("Progress", "")
                ).strip()

                if project and task:
                    triples.append({
                        "subject": project,
                        "relation": "has_task",
                        "object": task,
                        "source": workbook_name,
                        "modality": "TABLE"
                    })

                if task and person:
                    triples.append({
                        "subject": task,
                        "relation": "assigned_to",
                        "object": person,
                        "source": workbook_name,
                        "modality": "TABLE"
                    })

                if person and project:
                    triples.append({
                        "subject": person,
                        "relation": "works_on",
                        "object": project,
                        "source": workbook_name,
                        "modality": "TABLE"
                    })

                if task and progress:
                    triples.append({
                        "subject": task,
                        "relation": "progress",
                        "object": progress,
                        "source": workbook_name,
                        "modality": "TABLE"
                    })

        workbook.close()

    return triples