import pandas as pd

df = pd.read_csv('MASTER_ALL_WELLS_CORE_LOGS.csv')
html_table = df.to_html(classes='table table-striped table-bordered table-hover', index=False)

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>MASTER ALL WELLS CORE LOGS</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
    <style>
        body {{ padding: 25px; background-color: #f8f9fa; font-family: system-ui, -apple-system, sans-serif; }}
        h2 {{ color: #1e3a8a; font-weight: bold; margin-bottom: 5px; }}
        .card {{ border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); padding: 20px; background: white; }}
        table {{ font-size: 13px; }}
        th {{ background-color: #1e3a8a !important; color: white !important; font-weight: 600; text-align: center; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="card">
            <h2>🛢️ Master Core Logs Database (67 Records)</h2>
            <p class="text-muted">Interactive geological table — sort by clicking headers, search by well or operator, and filter instantly.</p>
            {html_table}
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    <script>
        $(document).ready(function() {{
            $('table').DataTable({{
                pageLength: 15,
                scrollX: true,
                order: [[0, 'asc'], [14, 'asc']]
            }});
        }});
    </script>
</body>
</html>"""

with open("view_csv_table.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Generated clean view_csv_table.html successfully!")

