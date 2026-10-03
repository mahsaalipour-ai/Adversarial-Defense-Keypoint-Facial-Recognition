import glob, os, csv
rows = []
for p in glob.glob("data/defense_out/*_squeeze.txt"):
    name = os.path.basename(p).replace("_squeeze.txt","")
    with open(p, encoding="utf-8") as f:
        txt = f.read()
    max_div = None
    flagged = None
    for line in txt.splitlines():
        if line.strip().startswith("max_div:"):
            max_div = float(line.split(":")[1].strip())
        if line.strip().startswith("flagged_adv:"):
            flagged = int(line.split(":")[1].strip())
    rows.append((name, max_div, flagged, p))
os.makedirs("data/defense_stats", exist_ok=True)
with open("data/defense_stats/squeeze_summary.csv","w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["name","max_div","flagged","report_path"])
    w.writerows(rows)
print("wrote data/defense_stats/squeeze_summary.csv, rows:", len(rows))
