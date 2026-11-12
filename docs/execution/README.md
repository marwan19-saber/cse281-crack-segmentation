# Planning documents (sources)

The PDFs are printed from these HTML files with a headless Chromium browser.
Run from the project root after editing any `.html` or `style.css` file.

Windows (Edge):

    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless=new --no-pdf-header-footer --print-to-pdf="$PWD/CSE281_Team_Execution_Plan.pdf" "file:///$PWD/docs/execution/CSE281_Team_Execution_Plan.html"

Mac / Linux (Chrome or Chromium): same flags, with `google-chrome` or `chromium` as the program.

| Source | Output |
|---|---|
| `CSE281_Team_Execution_Plan.html` + `style.css` | `CSE281_Team_Execution_Plan.pdf` (project root) |
| `report_draft.html` | `report/report_draft.pdf` |
| `presentation_draft.html` | `report/presentation_draft.pdf` |
