"""Build Highlander trading calculator + journal Excel workbook."""
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles.differential import DifferentialStyle
from openpyxl.utils import get_column_letter

OUTPUT = "Highlander_Trading_Calculator.xlsx"

# ── Colours ───────────────────────────────────────────────────────────────────
NAVY    = "1B2A4A"
BLUE    = "2E5090"
LBLUE   = "D6E4F7"
INPUT   = "FFF9E6"
CALC    = "EBF3FF"
LGRAY   = "F2F2F2"
DGRAY   = "BFBFBF"
WHITE   = "FFFFFF"
WIN_BG  = "E2EFDA"
LOSS_BG = "FCE4D6"
BE_BG   = "FFEB9C"

def fill(hex_):
    return PatternFill("solid", fgColor=hex_)

def font(bold=False, color=None, size=11):
    return Font(bold=bold, color=color or "000000", size=size)

def border(sides="all"):
    thin = Side(style="thin", color=DGRAY)
    none = Side(style=None)
    if sides == "all":
        return Border(left=thin, right=thin, top=thin, bottom=thin)
    if sides == "bottom":
        return Border(bottom=thin)
    return Border()

def centre():
    return Alignment(horizontal="center", vertical="center", wrap_text=True)

def right():
    return Alignment(horizontal="right", vertical="center")

def left():
    return Alignment(horizontal="left", vertical="center")

def hdr_cell(ws, row, col, text, bg=NAVY, fg=WHITE, bold=True, size=11, span=1):
    cell = ws.cell(row=row, column=col, value=text)
    cell.fill    = fill(bg)
    cell.font    = Font(bold=bold, color=fg, size=size)
    cell.alignment = centre()
    cell.border  = border()
    if span > 1:
        ws.merge_cells(start_row=row, start_column=col,
                       end_row=row, end_column=col+span-1)
    return cell

def label_cell(ws, row, col, text):
    cell = ws.cell(row=row, column=col, value=text)
    cell.fill      = fill(LGRAY)
    cell.font      = font(bold=False, size=10)
    cell.alignment = left()
    cell.border    = border()
    return cell

def input_cell(ws, row, col, value=None, fmt=None):
    cell = ws.cell(row=row, column=col, value=value)
    cell.fill      = fill(INPUT)
    cell.font      = font(size=11)
    cell.alignment = centre()
    cell.border    = border()
    if fmt:
        cell.number_format = fmt
    return cell

def calc_cell(ws, row, col, formula, fmt=None):
    cell = ws.cell(row=row, column=col, value=formula)
    cell.fill      = fill(CALC)
    cell.font      = font(size=11)
    cell.alignment = centre()
    cell.border    = border()
    if fmt:
        cell.number_format = fmt
    return cell


# ══════════════════════════════════════════════════════════════════════════════
#  Sheet 1 — Calculator
# ══════════════════════════════════════════════════════════════════════════════
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Calculator"

# Column widths
ws.column_dimensions["A"].width = 26
ws.column_dimensions["B"].width = 18
ws.column_dimensions["C"].width = 18
ws.column_dimensions["D"].width = 18
ws.column_dimensions["E"].width = 5   # spacer

# ── Title ─────────────────────────────────────────────────────────────────────
ws.row_dimensions[1].height = 30
hdr_cell(ws, 1, 1, "HIGHLANDER POSITION CALCULATOR", bg=NAVY, fg=WHITE, bold=True, size=14, span=4)

# ── Account Setup ─────────────────────────────────────────────────────────────
ws.row_dimensions[2].height = 8
hdr_cell(ws, 3, 1, "ACCOUNT SETUP", bg=BLUE, fg=WHITE, bold=True, size=10, span=4)

setup = [
    (4,  "Total Account Balance",  13000,     '"$"#,##0.00', False),
    (5,  "Risk % per Trade",        0.01,     '0.0%',        False),
    (6,  "Risk $ per Trade",       "=$B$4*$B$5", '"$"#,##0.00', True),
    (7,  "Per-Slot Margin (÷ 3)",  "=$B$4/3",  '"$"#,##0.00', True),
]
for row, lbl, val, fmt, is_calc in setup:
    label_cell(ws, row, 1, lbl)
    if is_calc:
        calc_cell(ws, row, 2, val, fmt)
        for c in [3, 4]:
            ws.cell(row=row, column=c).fill = fill(LGRAY)
            ws.cell(row=row, column=c).border = border()
    else:
        inp = input_cell(ws, row, 2, val, fmt)
        for c in [3, 4]:
            ws.cell(row=row, column=c).fill = fill(LGRAY)
            ws.cell(row=row, column=c).border = border()

# ── Notes on risk ─────────────────────────────────────────────────────────────
ws.row_dimensions[8].height = 8
note = ws.cell(row=9, column=1, value="ℹ  Risk options: 1% = conservative  |  1.5% = standard  |  2% = aggressive  |  2.5% = max")
note.font = Font(size=9, color="595959", italic=True)
note.alignment = left()
ws.merge_cells("A9:D9")

# ── Trade column headers ──────────────────────────────────────────────────────
ws.row_dimensions[10].height = 8
ws.row_dimensions[11].height = 22
hdr_cell(ws, 11, 1, "FIELD",    bg=NAVY, fg=WHITE, size=10)
hdr_cell(ws, 11, 2, "TRADE 1",  bg=NAVY, fg=WHITE, size=11)
hdr_cell(ws, 11, 3, "TRADE 2",  bg=NAVY, fg=WHITE, size=11)
hdr_cell(ws, 11, 4, "TRADE 3",  bg=NAVY, fg=WHITE, size=11)

# ── Input rows ────────────────────────────────────────────────────────────────
INPUT_ROWS = [
    (12, "Symbol",          None,     "@",         False),
    (13, "Direction",       "Long",   "@",         False),
    (14, "Entry Price",     None,     '#,##0.0000', False),
    (15, "Stop Price",      None,     '#,##0.0000', False),
    (16, "TP (2.618 auto)", None,     '#,##0.0000', True),  # calculated
]

dir_dv = DataValidation(type="list", formula1='"Long,Short"', allow_blank=True)
dir_dv.sqref = "B13:D13"
ws.add_data_validation(dir_dv)

# TP formula: entry ± |entry-stop| × (1.618/1.5)
def tp_formula(col_letter):
    return (f'=IF({col_letter}14="","",IF({col_letter}13="Long",'
            f'{col_letter}14+ABS({col_letter}14-{col_letter}15)*(1.618/1.5),'
            f'IF({col_letter}13="Short",'
            f'{col_letter}14-ABS({col_letter}14-{col_letter}15)*(1.618/1.5),"")))')

for row, lbl, default, fmt, is_calc in INPUT_ROWS:
    ws.row_dimensions[row].height = 20
    label_cell(ws, row, 1, lbl)
    for col, col_letter in [(2, "B"), (3, "C"), (4, "D")]:
        if is_calc:
            calc_cell(ws, row, col, tp_formula(col_letter), fmt)
        else:
            input_cell(ws, row, col, default if col == 2 else None, fmt)

# ── Divider ───────────────────────────────────────────────────────────────────
ws.row_dimensions[17].height = 20
hdr_cell(ws, 17, 1, "CALCULATED OUTPUTS", bg=BLUE, fg=WHITE, size=10, span=4)

# ── Calculated output rows ────────────────────────────────────────────────────
def formula(col_letter, row_formula):
    return f'=IF({col_letter}14="","",{row_formula})'

CALC_ROWS = []
for cl in ["B", "C", "D"]:
    e, s, tp = f"{cl}14", f"{cl}15", f"{cl}16"
    sd = f"ABS({e}-{s})"   # stop distance $
    # lev = leverage needed = CEILING(position_size / per_slot_margin, 1)
    # position_size_$ = units * entry = (risk$ / stop_dist) * entry
    lev  = f"MAX(1,CEILING($B$6/{sd}*{e}/$B$7,1))"
    pos  = f"$B$6/{sd}*{e}"      # position size $  = units × entry
    uts  = f"$B$6/{sd}"          # units (coins)     = risk$ / stop_dist
    CALC_ROWS_COL = [
        (18, "Stop Distance ($)",        formula(cl, f"{sd}"),                    '"$"#,##0.0000'),
        (19, "Stop Distance (%)",        formula(cl, f"{sd}/{e}*100"),             '0.000"%"'),
        (20, "Position Size ($)",        formula(cl, f"{pos}"),                   '"$"#,##0.00'),
        (21, "Units (coins)",            formula(cl, f"{uts}"),                   '#,##0.00'),
        (22, "Leverage Needed",          formula(cl, f"{lev}"),                   '0"x"'),
        (23, "Margin Used ($)",          formula(cl, f"({pos})/{lev}"),           '"$"#,##0.00'),
        (24, "Free Margin ($)",          formula(cl, f"$B$7-({pos})/{lev}"),      '"$"#,##0.00'),
        (25, "Profit if TP ($)",         formula(cl, f"ABS({tp}-{e})*({uts})"),  '"$"#,##0.00'),
        (26, "Profit if TP (% acct)",    formula(cl, f"ABS({tp}-{e})*({uts})/$B$4*100"), '0.00"%"'),
        (27, "Actual R:R",               formula(cl, f"ABS({tp}-{e})/{sd}"),      '0.000":1"'),
        (28, "Max Loss ($)",             formula(cl, f"$B$6"),                    '"$"#,##0.00'),
        (29, "Max Loss (% acct)",        formula(cl, f"$B$5*100"),                '0.0"%"'),
    ]
    CALC_ROWS.append(CALC_ROWS_COL)

# Write labels and formulas
for row_idx, row_num in enumerate(range(18, 30)):
    ws.row_dimensions[row_num].height = 20
    lbl = CALC_ROWS[0][row_idx][0]  # label same for all cols
    label_cell(ws, row_num, 1, CALC_ROWS[0][row_idx][1])  # desc
    for col_offset, col_data in enumerate(CALC_ROWS):
        _, _, formula_val, fmt = col_data[row_idx]
        calc_cell(ws, row_num, 2 + col_offset, formula_val, fmt)

# ── Summary row ───────────────────────────────────────────────────────────────
ws.row_dimensions[30].height = 8
ws.row_dimensions[31].height = 22
hdr_cell(ws, 31, 1, "TOTAL EXPOSURE", bg=BLUE, fg=WHITE, size=10)
calc_cell(ws, 31, 2, '=IF(B14="","",B28)',  '"$"#,##0.00')
calc_cell(ws, 31, 3, '=IF(C14="","",C28)',  '"$"#,##0.00')
calc_cell(ws, 31, 4, '=IF(D14="","",D28)',  '"$"#,##0.00')

ws.row_dimensions[32].height = 22
hdr_cell(ws, 32, 1, "COMBINED RISK IF ALL OPEN", bg=NAVY, fg=WHITE, size=10)
calc_cell(ws, 32, 2,
    '=IFERROR(SUM(IF(B14<>"",B28,0),IF(C14<>"",C28,0),IF(D14<>"",D28,0)),"—")',
    '"$"#,##0.00')
ws.merge_cells("B32:D32")
ws.cell(32, 2).alignment = centre()

ws.row_dimensions[33].height = 22
hdr_cell(ws, 33, 1, "COMBINED RISK % OF ACCOUNT", bg=NAVY, fg=WHITE, size=10)
calc_cell(ws, 33, 2,
    '=IFERROR(SUM(IF(B14<>"",B29,0),IF(C14<>"",C29,0),IF(D14<>"",D29,0)),"—")',
    '0.0"%"')
ws.merge_cells("B33:D33")
ws.cell(33, 2).alignment = centre()

# ── Legend ────────────────────────────────────────────────────────────────────
ws.row_dimensions[35].height = 8
note2 = ws.cell(row=36, column=1,
    value="⚠  Leverage shown is minimum needed to hold the position in your allocated slot (balance ÷ 3)."
          "  Stop is always set BEFORE entry. TP = origin + 2.618 × range (actual RR ≈ 1.079:1).")
note2.font = Font(size=9, color="595959", italic=True)
note2.alignment = Alignment(horizontal="left", wrap_text=True)
ws.merge_cells("A36:D36")
ws.row_dimensions[36].height = 30


# ══════════════════════════════════════════════════════════════════════════════
#  Sheet 2 — Trade Journal
# ══════════════════════════════════════════════════════════════════════════════
wj = wb.create_sheet("Journal")

# Column widths
col_widths = {
    "A": 5,  "B": 13, "C": 13, "D": 8,  "E": 8,
    "F": 11, "G": 11, "H": 11, "I": 8,  "J": 10,
    "K": 13, "L": 10, "M": 8,  "N": 13,
    "O": 11, "P": 11, "Q": 10, "R": 14, "S": 14, "T": 14,
    "U": 20,
}
for col, w in col_widths.items():
    wj.column_dimensions[col].width = w

# Headers
wj.row_dimensions[1].height = 30
hdr_cell(wj, 1, 1, "HIGHLANDER TRADE JOURNAL", bg=NAVY, fg=WHITE, bold=True, size=14, span=21)

headers = [
    "#", "Date\nOpened", "Date\nClosed", "Symbol", "Direction",
    "Entry", "Stop", "TP", "Risk\n%", "Risk\n$",
    "Position\nSize $", "Units", "Lev",  "Outcome",
    "Exit\nPrice", "P&L $", "R\nMultiple", "Balance\nBefore", "Balance\nAfter",
    "Running\nP&L $", "Notes"
]
wj.row_dimensions[2].height = 36
for c, h in enumerate(headers, 1):
    cell = wj.cell(row=2, column=c, value=h)
    cell.fill      = fill(BLUE)
    cell.font      = Font(bold=True, color=WHITE, size=9)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border    = border()

# Outcome dropdown
outcome_dv = DataValidation(type="list",
    formula1='"Win,Full Stop,BE Stop,Open"', allow_blank=True)
outcome_dv.sqref = "N3:N202"
wj.add_data_validation(outcome_dv)

dir_dv2 = DataValidation(type="list", formula1='"Long,Short"', allow_blank=True)
dir_dv2.sqref = "E3:E202"
wj.add_data_validation(dir_dv2)

# Pre-fill 100 journal rows
for row in range(3, 103):
    wj.row_dimensions[row].height = 18
    r = row

    # Row number
    n = wj.cell(row=r, column=1, value=r-2)
    n.fill = fill(LGRAY); n.alignment = centre(); n.border = border(); n.font = font(size=9)

    # Input cells: B(date), C(date), D(sym), E(dir), F(entry), G(stop), H(tp), I(risk%), J(risk$)
    for c in range(2, 11):
        cell = wj.cell(row=r, column=c)
        cell.fill   = fill(INPUT)
        cell.border = border()
        cell.alignment = centre()
        cell.font = font(size=10)

    # Formats
    wj.cell(r, 2).number_format  = "DD-MMM-YY"
    wj.cell(r, 3).number_format  = "DD-MMM-YY"
    wj.cell(r, 6).number_format  = "#,##0.0000"
    wj.cell(r, 7).number_format  = "#,##0.0000"
    wj.cell(r, 8).number_format  = "#,##0.0000"
    wj.cell(r, 9).number_format  = "0.0%"
    wj.cell(r, 10).number_format = '"$"#,##0.00'

    # Calculated: K(pos size), L(units), M(lev), N(outcome-input), O(exit-input),
    #             P(P&L), Q(R mult), R(bal before-input), S(bal after), T(running)
    # K: position size $ = (risk$ / stop_dist) * entry
    k = wj.cell(r, 11, value=f'=IF(F{r}="","",J{r}/ABS(F{r}-G{r})*F{r})')
    k.fill = fill(CALC); k.border = border(); k.alignment = centre()
    k.number_format = '"$"#,##0.00'; k.font = font(size=10)

    # L: units = pos_size / entry
    l = wj.cell(r, 12, value=f'=IF(F{r}="","",K{r}/F{r})')
    l.fill = fill(CALC); l.border = border(); l.alignment = centre()
    l.number_format = '#,##0.000'; l.font = font(size=10)

    # M: leverage = CEILING(pos_size / (balance_before/3), 1)
    m = wj.cell(r, 13, value=f'=IF(F{r}="","",MAX(1,CEILING(K{r}/(R{r}/3),1)))')
    m.fill = fill(CALC); m.border = border(); m.alignment = centre()
    m.number_format = '0"x"'; m.font = font(size=10)

    # N: outcome (input)
    n2 = wj.cell(r, 14)
    n2.fill = fill(INPUT); n2.border = border(); n2.alignment = centre(); n2.font = font(size=10)

    # O: exit price (input)
    o = wj.cell(r, 15)
    o.fill = fill(INPUT); o.border = border(); o.alignment = centre()
    o.number_format = "#,##0.0000"; o.font = font(size=10)

    # P: P&L $ = (exit - entry) * units for long, reversed for short
    p = wj.cell(r, 16,
        value=f'=IF(OR(F{r}="",O{r}=""),"",IF(E{r}="Long",(O{r}-F{r})*L{r},(F{r}-O{r})*L{r}))')
    p.fill = fill(CALC); p.border = border(); p.alignment = centre()
    p.number_format = '"$"#,##0.00'; p.font = font(size=10)

    # Q: R multiple = P&L / risk$
    q = wj.cell(r, 17,
        value=f'=IF(OR(J{r}="",J{r}=0,P{r}=""),"",P{r}/J{r})')
    q.fill = fill(CALC); q.border = border(); q.alignment = centre()
    q.number_format = '0.00"R"'; q.font = font(size=10)

    # R: balance before (input)
    rb = wj.cell(r, 18)
    rb.fill = fill(INPUT); rb.border = border(); rb.alignment = centre()
    rb.number_format = '"$"#,##0.00'; rb.font = font(size=10)

    # S: balance after = balance before + P&L
    s = wj.cell(r, 19,
        value=f'=IF(OR(R{r}="",P{r}=""),"",R{r}+P{r})')
    s.fill = fill(CALC); s.border = border(); s.alignment = centre()
    s.number_format = '"$"#,##0.00'; s.font = font(size=10)

    # T: running P&L = SUM of all P&L up to this row
    t = wj.cell(r, 20,
        value=f'=IF(P{r}="","",SUM(P$3:P{r}))')
    t.fill = fill(CALC); t.border = border(); t.alignment = centre()
    t.number_format = '"$"#,##0.00'; t.font = font(size=10)

    # U: notes (input)
    u = wj.cell(r, 21)
    u.fill = fill(INPUT); u.border = border()
    u.alignment = Alignment(horizontal="left", vertical="center")
    u.font = font(size=10)

# Conditional formatting — colour rows by outcome
from openpyxl.formatting.rule import FormulaRule

green_fill  = PatternFill("solid", fgColor=WIN_BG)
red_fill    = PatternFill("solid", fgColor=LOSS_BG)
yellow_fill = PatternFill("solid", fgColor=BE_BG)

for row in range(3, 103):
    rng = f"A{row}:U{row}"
    wj.conditional_formatting.add(rng,
        FormulaRule(formula=[f'$N{row}="Win"'],       fill=green_fill))
    wj.conditional_formatting.add(rng,
        FormulaRule(formula=[f'$N{row}="Full Stop"'], fill=red_fill))
    wj.conditional_formatting.add(rng,
        FormulaRule(formula=[f'$N{row}="BE Stop"'],   fill=yellow_fill))

# ── Journal Summary block ─────────────────────────────────────────────────────
wj.row_dimensions[104].height = 8
hdr_cell(wj, 105, 1, "SUMMARY", bg=NAVY, fg=WHITE, size=11, span=6)

summary = [
    (106, "Total Trades",  '=COUNTA(N3:N102)-COUNTIF(N3:N102,"Open")'),
    (107, "Wins",          '=COUNTIF(N3:N102,"Win")'),
    (108, "Full Stops",    '=COUNTIF(N3:N102,"Full Stop")'),
    (109, "BE Stops",      '=COUNTIF(N3:N102,"BE Stop")'),
    (110, "Win Rate",      '=IF(B106=0,"",B107/B106)'),
    (111, "Total R",       '=IF(COUNTA(Q3:Q102)=0,"",SUMIF(Q3:Q102,"<>",Q3:Q102))'),
    (112, "Avg R/trade",   '=IF(B106=0,"",B111/B106)'),
    (113, "Profit Factor", '=IF(SUMIF(Q3:Q102,"<0",Q3:Q102)=0,"",SUMIF(Q3:Q102,">0",Q3:Q102)/ABS(SUMIF(Q3:Q102,"<0",Q3:Q102)))'),
    (114, "Total P&L $",   '=IF(COUNTA(P3:P102)=0,"",SUM(P3:P102))'),
]
fmts = ["0", "0", "0", "0", "0.0%", "0.00", "0.000", "0.00", '"$"#,##0.00']
for i, (row, lbl, form) in enumerate(summary):
    wj.row_dimensions[row].height = 20
    label_cell(wj, row, 1, lbl)
    c = wj.cell(row=row, column=2, value=form)
    c.fill = fill(CALC); c.border = border(); c.alignment = centre()
    c.number_format = fmts[i]; c.font = font(size=11)
    for col in range(3, 7):
        wj.cell(row=row, column=col).fill = fill(LGRAY)
        wj.cell(row=row, column=col).border = border()

# ── Freeze panes ─────────────────────────────────────────────────────────────
ws.freeze_panes  = "B12"
wj.freeze_panes = "A3"

# ══════════════════════════════════════════════════════════════════════════════
#  Sheet 3 — Strategy Rules
# ══════════════════════════════════════════════════════════════════════════════
wr = wb.create_sheet("Rules")
wr.column_dimensions["A"].width = 3    # left margin
wr.column_dimensions["B"].width = 28   # label/icon column
wr.column_dimensions["C"].width = 62   # content column
wr.column_dimensions["D"].width = 3    # right margin

def rules_title(ws, row, text, bg=NAVY, span_cols="B:C"):
    ws.row_dimensions[row].height = 28
    cell = ws.cell(row=row, column=2, value=text)
    cell.fill      = fill(bg)
    cell.font      = Font(bold=True, color=WHITE, size=13)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.border    = border()
    ws.merge_cells(f"B{row}:C{row}")
    return cell

def rules_section(ws, row, text, bg=BLUE):
    ws.row_dimensions[row].height = 22
    cell = ws.cell(row=row, column=2, value=text)
    cell.fill      = fill(bg)
    cell.font      = Font(bold=True, color=WHITE, size=11)
    cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    cell.border    = border()
    ws.merge_cells(f"B{row}:C{row}")

def rules_row(ws, row, label, content, label_bg=LGRAY, content_bg=WHITE, bold_label=True, h=18):
    ws.row_dimensions[row].height = h
    lc = ws.cell(row=row, column=2, value=label)
    lc.fill      = fill(label_bg)
    lc.font      = Font(bold=bold_label, size=10, color="1B2A4A")
    lc.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    lc.border    = border()
    cc = ws.cell(row=row, column=3, value=content)
    cc.fill      = fill(content_bg)
    cc.font      = Font(size=10)
    cc.alignment = Alignment(horizontal="left", vertical="center", indent=1, wrap_text=True)
    cc.border    = border()

def rules_note(ws, row, text, bg="FFF3CD", text_color="7D4E00"):
    ws.row_dimensions[row].height = 20
    cell = ws.cell(row=row, column=2, value=text)
    cell.fill      = fill(bg)
    cell.font      = Font(italic=True, size=9, color=text_color)
    cell.alignment = Alignment(horizontal="left", vertical="center", indent=1, wrap_text=True)
    cell.border    = border()
    ws.merge_cells(f"B{row}:C{row}")

def blank_row(ws, row, h=6):
    ws.row_dimensions[row].height = h
    for c in [2, 3]:
        ws.cell(row=row, column=c).fill = fill(WHITE)

r = 1
# ── Main title ────────────────────────────────────────────────────────────────
rules_title(wr, r, "HIGHLANDER STRATEGY — RULES & PLAYBOOK"); r += 1
blank_row(wr, r); r += 1

# ── Section 1: How it works ───────────────────────────────────────────────────
rules_section(wr, r, "1.  HOW THE STRATEGY WORKS"); r += 1
rules_row(wr, r, "Setup name",        "Highlander Engulfing — fib retest at a support/resistance level"); r += 1
rules_row(wr, r, "Timeframe",         "1H candles  (signal bar is 1 hour)"); r += 1
rules_row(wr, r, "Symbols traded",    "SOL/USDT  and  SUI/USDT  (monitored simultaneously by bot)"); r += 1
rules_row(wr, r, "Edge",              "Engulfing candle at a Highlander level → price retests → ML model confirms → enter"); r += 1
rules_row(wr, r, "Win rate (OOS)",    "SOL 56.6%  |  SUI 54.4%  (out-of-sample walk-forward validation)"); r += 1
rules_row(wr, r, "Profit factor",     "SOL 2.76  |  SUI 2.20  (after costs, actual 1.079:1 RR)"); r += 1
rules_row(wr, r, "Actual R:R",        "1.079:1  — stop is 1.5× range below entry, TP is 1.618× range above entry"); r += 1
blank_row(wr, r); r += 1

# ── Section 2: Signal rules ───────────────────────────────────────────────────
rules_section(wr, r, "2.  SIGNAL — WHAT THE BOT LOOKS FOR  (you don't need to do this manually)"); r += 1
rules_row(wr, r, "Step 1 — Engulfing", "A bullish or bearish engulfing candle forms on the 1H chart"); r += 1
rules_row(wr, r, "Step 2 — Level",     "The engulfing candle must be at (or near) an active Highlander support/resistance level"); r += 1
rules_row(wr, r, "Step 3 — ML filter", "CatBoost model scores the setup.  Score ≥ 0.50 = valid signal.  Below = filtered out (shown in /filtered)"); r += 1
rules_row(wr, r, "Step 4 — Alert",     "Bot sends Discord push notification with entry, stop and TP prices"); r += 1
rules_note(wr, r, "⚠  If you miss the Discord alert, run /pending in Discord to see waiting setups.  Run /filtered to see what the model rejected."); r += 1
blank_row(wr, r); r += 1

# ── Section 3: Fib levels explained ──────────────────────────────────────────
rules_section(wr, r, "3.  FIB LEVELS — THE TRADE STRUCTURE"); r += 1
rules_row(wr, r, "Origin  (fib 0.0)",   "Low of pre-engulfing candle (bull)  /  High (bear).  This is where the level was created."); r += 1
rules_row(wr, r, "Entry   (fib 1.0)",   "High of pre-engulfing candle (bull)  /  Low (bear).  Price must RETRACE here to activate."); r += 1
rules_row(wr, r, "Stop    (fib −0.5)",  "Entry − 1.5 × range  (bull)  /  Entry + 1.5 × range  (bear).  Set this BEFORE entering."); r += 1
rules_row(wr, r, "BE level (fib 1.618)","Entry + 1.618 × range  (bull).  When price TOUCHES this → move stop to entry immediately."); r += 1
rules_row(wr, r, "Take Profit (fib 2.618)", "Entry + 1.618/1.5 × stop_dist  (bull).  Auto-calculated in the Excel calculator."); r += 1
rules_note(wr, r, "ℹ  The TP label says '2.618R' in backtests but actual RR is 1.079:1 because stop distance = 1.5 × range, not 1× range."); r += 1
blank_row(wr, r); r += 1

# ── Section 4: Entry rules ────────────────────────────────────────────────────
rules_section(wr, r, "4.  ENTRY RULES"); r += 1
rules_row(wr, r, "When to enter",       "ONLY after bot alert (Discord notification or /pending command shows the setup)"); r += 1
rules_row(wr, r, "Entry price",         "Use the EXACT entry price from the bot — do not guess or use a different level"); r += 1
rules_row(wr, r, "Order type",          "Limit order at the entry price.  Do not use market orders."); r += 1
rules_row(wr, r, "Expiry",              "If price has not retested entry within 50 bars (~50 hours) → cancel the order.  Bot marks it as Cancelled."); r += 1
rules_row(wr, r, "Max open trades",     "3 maximum simultaneously.  Never open a 4th until one closes."); r += 1
rules_row(wr, r, "Same symbol",         "If 2 setups appear on SOL at the same time, only take the one with the higher ML score"); r += 1
rules_note(wr, r, "⚠  NEVER chase price.  If price blows through the entry level without a clean retest, skip the trade."); r += 1
blank_row(wr, r); r += 1

# ── Section 5: Position sizing ────────────────────────────────────────────────
rules_section(wr, r, "5.  POSITION SIZING  (use the Calculator sheet)"); r += 1
rules_row(wr, r, "Risk per trade",      "1.0%–1.5% of TOTAL account balance (not per-slot balance)"); r += 1
rules_row(wr, r, "Account split",       "Account ÷ 3 = margin available per open trade slot"); r += 1
rules_row(wr, r, "Units formula",       "Units = Risk$ ÷ Stop_distance_$     e.g. $130 ÷ $0.80 = 162.5 SOL"); r += 1
rules_row(wr, r, "Position size",       "Position$ = Units × Entry price     e.g. 162.5 × $89.70 = $14,576"); r += 1
rules_row(wr, r, "Leverage",            "Leverage = CEILING(Position$ ÷ Slot_margin, 1)     e.g. $14,576 ÷ $4,333 = 4x"); r += 1
rules_row(wr, r, "Leverage limit",      "Never go above 10x.  If the calculator shows >10x, reduce risk % or skip the trade."); r += 1
rules_note(wr, r, "ℹ  Leverage does NOT increase your risk %.  Your risk is always fixed by the stop loss distance.  Leverage only reduces margin required."); r += 1
blank_row(wr, r); r += 1

# ── Section 6: Stop loss management ──────────────────────────────────────────
rules_section(wr, r, "6.  STOP LOSS MANAGEMENT  — THE MOST IMPORTANT RULE"); r += 1
rules_row(wr, r, "Set stop immediately",  "As soon as your limit order fills → set stop loss at the fib −0.5 level.  No exceptions.", content_bg="FCE4D6"); r += 1
rules_row(wr, r, "NEVER widen the stop", "If price approaches your stop — do nothing.  Let it hit.  Widening stops destroys the edge.", content_bg="FCE4D6"); r += 1
rules_row(wr, r, "Move to break-even",   "When price hits fib 1.618 (BE level) → immediately move stop to ENTRY price.  Do this manually on Bybit.", content_bg=WIN_BG); r += 1
rules_row(wr, r, "After moving to BE",   "Worst case is now ~−0.04R (just costs).  Let the trade run to TP at 2.618."); r += 1
rules_row(wr, r, "Bot tracking",         "Bot automatically tracks BE stops.  A 'BE Stop' result = small loss of ~−$5 (costs only)."); r += 1
rules_note(wr, r, "⚠  The BE stop rule is what gives this strategy its edge.  Skipping it turns 1.079:1 RR into a losing system."); r += 1
blank_row(wr, r); r += 1

# ── Section 7: Take profit ────────────────────────────────────────────────────
rules_section(wr, r, "7.  TAKE PROFIT"); r += 1
rules_row(wr, r, "TP target",           "fib 2.618  =  Entry + 1.618/1.5 × Stop_distance  (auto-calculated in Excel)"); r += 1
rules_row(wr, r, "Order type",          "Set a limit take-profit order at the TP price immediately after entry fills"); r += 1
rules_row(wr, r, "Partial TP",          "Optional: take 50% at fib 1.618 (BE level) and let rest run to 2.618.  Default: full TP at 2.618 only."); r += 1
rules_row(wr, r, "If TP not reached",   "After 200 bars (~8 days) without resolution → close manually at market.  Bot flags this."); r += 1
blank_row(wr, r); r += 1

# ── Section 8: Discord commands ──────────────────────────────────────────────
rules_section(wr, r, "8.  DISCORD BOT COMMANDS"); r += 1
rules_row(wr, r, "/overview",   "Full status — pending setups, active trades, recent results, filtered, all symbols"); r += 1
rules_row(wr, r, "/pending",    "Setups waiting for price to retest the entry level (not yet activated)"); r += 1
rules_row(wr, r, "/active",     "Trades currently open — shows unrealised R"); r += 1
rules_row(wr, r, "/results",    "Recent completed trades — won, full stop, BE stop"); r += 1
rules_row(wr, r, "/filtered",   "Setups that retested but ML model scored < 0.50 — do NOT take these manually"); r += 1
rules_row(wr, r, "/cancelled",  "Setups that expired without a retest (price never came back within 50 bars)"); r += 1
rules_note(wr, r, "ℹ  Bot runs live_signals.py — check it's running with:  ps aux | grep live_signals.  Restart if needed."); r += 1
blank_row(wr, r); r += 1

# ── Section 9: Key numbers ────────────────────────────────────────────────────
rules_section(wr, r, "9.  KEY NUMBERS AT A GLANCE"); r += 1
rules_row(wr, r, "Actual R:R",          "1.079 : 1   (not 2.618 — that's the fib label, not the real ratio)"); r += 1
rules_row(wr, r, "SOL signals/month",   "~24   |  Win rate 56.6%  |  Profit factor 2.76  |  Net 9.3R/month"); r += 1
rules_row(wr, r, "SUI signals/month",   "~33   |  Win rate 54.4%  |  Profit factor 2.20  |  Net 10.5R/month"); r += 1
rules_row(wr, r, "Combined R/month",    "~19.8R   →   19.8% monthly return at 1% risk per trade"); r += 1
rules_row(wr, r, "Year 1 at 1% risk",   "$13,000  →  ~$113,000  (compounding, average monthly returns)"); r += 1
rules_row(wr, r, "Year 1 at 1.5% risk", "$13,000  →  ~$295,000  (compounding — higher variance, more drawdown risk)"); r += 1
rules_note(wr, r, "⚠  These are backtest averages.  Real months vary widely.  Stick to rules in losing months — the edge is real."); r += 1
blank_row(wr, r); r += 1

# ── Section 10: Common mistakes ───────────────────────────────────────────────
rules_section(wr, r, "10.  COMMON MISTAKES — DO NOT DO THESE", bg="8B0000"); r += 1
rules_row(wr, r, "❌  Cherry-picking",   "Taking only signals you 'like' the look of.  Take every signal the bot passes.", content_bg=LOSS_BG); r += 1
rules_row(wr, r, "❌  Widening stops",   "Moving stop further away because 'it looks like it'll bounce'.  Never.", content_bg=LOSS_BG); r += 1
rules_row(wr, r, "❌  Moving TP early",  "Closing before TP because you've got profit.  Trust the system.", content_bg=LOSS_BG); r += 1
rules_row(wr, r, "❌  Skipping BE move", "Forgetting to move stop to entry when 1.618 hits.  Set a price alert on Bybit.", content_bg=LOSS_BG); r += 1
rules_row(wr, r, "❌  Oversizing",       "Risking more than 2.5% on a single trade.  Use the calculator every time.", content_bg=LOSS_BG); r += 1
rules_row(wr, r, "❌  Taking filtered",  "Manually entering setups the ML model rejected (/filtered list).  It filtered them for a reason.", content_bg=LOSS_BG); r += 1
rules_row(wr, r, "❌  4th open trade",   "Opening a trade when 3 are already active.  Wait for one to close.", content_bg=LOSS_BG); r += 1

# ── Tab colours ───────────────────────────────────────────────────────────────
ws.sheet_properties.tabColor  = "1B2A4A"
wj.sheet_properties.tabColor  = "2E5090"
wr.sheet_properties.tabColor  = "8B0000"

wb.save(OUTPUT)
print(f"Saved: {OUTPUT}")
