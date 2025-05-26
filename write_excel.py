from openpyxl import load_workbook, Workbook

filename = "validation_values.xlsx"

try:
    wb = load_workbook(filename)
    ws = wb.active
except FileNotFoundError:
    wb = Workbook()
    ws = wb.active
    ws.append(["CODE", "PRECISION", "RECALL","F1", "MEANIOU", "TIMEFLAT", "TIMEFLATNESTED" "TIMENESTED", "TYPE"])

code = CODE
precision = PRECISION
recall = RECALL
f1 = F1
iou = MEANIOU
timeflat = TIMEFLAT 
timeflatnested = TIMEFLATNESTED
timenested = TIMENESTED
type_ = TYPE 

already_exists = False
for row in ws.iter_rows(min_row=2, values_only=True):
    code_cell = row[0]
    type_cell = row[8]
    if code_cell == code and type_cell == type_:
        already_exists = True
        break

if not already_exists:
    ws.append([code, precision, recall, f1, iou, timeflat, timeflatnested, timenested, type_])

wb.save(filename)
