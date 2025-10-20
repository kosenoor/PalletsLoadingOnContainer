import os, random
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, session
from werkzeug.utils import secure_filename
from dataclasses import dataclass
from typing import List
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ------------------ CONFIG ------------------
UPLOAD_FOLDER = "uploads"
LAYOUT_IMAGE_DIR = "static/layout_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LAYOUT_IMAGE_DIR, exist_ok=True)

STACK_MAX = 0
SAVE_DPI  = 150
SECRET_KEY = os.urandom(24)  # for session

# ------------------ FLASK APP ------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

# ------------------ DATA CLASSES ------------------
@dataclass
class PalletSpec:
    pid: str; l: float; w: float; h: float; qty: int

@dataclass
class ContainerSpec:
    id: str; l: float; w: float; h: float; qty: int; type: str = ""

@dataclass
class LoadedItem:
    container_id: str; pallet_id: str
    x: float; y: float; z: float
    l: float; w: float; h: float
    row: int; col: int; layer: int
    qty_in_stack: int; label: str; stacked: bool

@dataclass
class FreeBox:
    x: float; y: float; z: float
    l: float; w: float; h: float
    def volume(self) -> float: return max(self.l,0) * max(self.w,0) * max(self.h,0)

# ------------------ ORIENTATIONS ------------------
def unique_orientations(p: PalletSpec):
    seen, out = set(), []
    dims = (p.l, p.w, p.h)
    candidates = [
        (dims[0],dims[1],dims[2]), (dims[1],dims[0],dims[2]),
        (dims[0],dims[2],dims[1]), (dims[2],dims[0],dims[1]),
        (dims[1],dims[2],dims[0]), (dims[2],dims[1],dims[0])
    ]
    for c in candidates:
        key = tuple(round(v,3) for v in c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out

# ------------------ HELPERS ------------------
def _pid(p): return p.pid if hasattr(p,"pid") else str(p)
def _overlap_1d(a0,a1,b0,b1): return not (a1 <= b0 or b1 <= a0)
def _intersects(a:LoadedItem,b:LoadedItem)->bool:
    return (_overlap_1d(a.x,a.x+a.l,b.x,b.x+b.l) and
            _overlap_1d(a.y,a.y+a.w,b.y,b.y+b.w) and
            _overlap_1d(a.z,a.z+a.h,b.z,b.z+b.h))

def split_free_both(box:FreeBox,item:LoadedItem)->List[FreeBox]:
    new_boxes=[]
    right_l=(box.x+box.l)-(item.x+item.l)
    if right_l>0: new_boxes.append(FreeBox(item.x+item.l,box.y,box.z,right_l,box.w,box.h))
    front_w=(box.y+box.w)-(item.y+item.w)
    if front_w>0: new_boxes.append(FreeBox(box.x,item.y+item.w,box.z,box.l,front_w,box.h))
    above_h=(box.z+box.h)-(item.z+item.h)
    if above_h>0: new_boxes.append(FreeBox(box.x,box.y,item.z+item.h,box.l,box.w,above_h))
    return [b for b in new_boxes if b.volume()>0]

# ------------------ PACK LOGIC ------------------
def pack(containers:List[ContainerSpec], pallets:List[PalletSpec]):
    loaded=[]; missed={_pid(p):int(p.qty) for p in pallets}
    total_left=sum(missed.values())
    pallets_sorted=sorted(pallets,key=lambda p:-(p.l*p.w*p.h))
    container_instances=[]
    for c in containers:
        for i in range(1,int(c.qty)+1):
            container_instances.append((c,f"{c.id}-{i}"))
    container_instances.sort(key=lambda x:-(x[0].l*x[0].w*x[0].h))
    for c,cid in container_instances:
        if total_left<=0: break
        free=[FreeBox(0,0,0,c.l,c.w,c.h)]
        row=1;col=1;placed_any=True
        while placed_any and total_left>0:
            placed_any=False; free.sort(key=lambda b:(b.z,b.y,b.x))
            i=0
            while i<len(free) and total_left>0:
                fb=free[i]; best=None; best_util=0
                for pal in pallets_sorted:
                    pid=_pid(pal); req=missed[pid]
                    if req<=0: continue
                    for (l,w,h) in unique_orientations(pal):
                        if l<=fb.l and w<=fb.w and h<=fb.h:
                            max_by_height=int(fb.h//h)
                            stacks=min(max_by_height,req,STACK_MAX or max_by_height)
                            if stacks<=0: continue
                            util=(l*w*h*stacks)/fb.volume()
                            if util>best_util:
                                best_util=util; best=(pal,pid,l,w,h,stacks)
                if best:
                    pal,pid,l,w,h,stacks=best; consumed_h=h*stacks
                    candidate=LoadedItem(
                        cid,pid,fb.x,fb.y,fb.z,
                        l,w,consumed_h,row,col,int(fb.z//max(h,1))+1,
                        stacks,pid,stacks>1)
                    collides=any(_intersects(candidate,ex) for ex in loaded if ex.container_id==cid and abs(ex.z-candidate.z)<max(h,ex.h))
                    if collides: i+=1; continue
                    loaded.append(candidate); missed[pid]-=stacks; total_left-=stacks
                    free.pop(i); free.extend(split_free_both(fb,candidate))
                    placed_any=True; col+=1
                else: i+=1
            if placed_any: row+=1;col=1
    return loaded,missed

# ------------------ DRAW ------------------
import matplotlib.colors as mcolors
def draw_2d_all(loaded, containers, folder):
    if not loaded: return []
    os.makedirs(folder, exist_ok=True)
    ids = sorted({li.container_id for li in loaded})
    files = []

    # Assign unique color to each pallet ID
    unique_pallets = sorted({li.pallet_id for li in loaded})
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    pallet_colors = {pid: colors[i % len(colors)] for i, pid in enumerate(unique_pallets)}

    for cid in ids:
        items = [li for li in loaded if li.container_id == cid]
        c = next(cc for cc in containers if cid.startswith(cc.id))
        fig, ax = plt.subplots(figsize=(12,6))
        ax.add_patch(Rectangle((0,0), c.l, c.w, linewidth=2, edgecolor="black", facecolor="none"))
        for li in items:
            rect = Rectangle((li.x, li.y), li.l, li.w, edgecolor="black", facecolor=pallet_colors[li.pallet_id], alpha=0.8)
            ax.add_patch(rect)
            ax.text(li.x+li.l/2, li.y+li.w/2, f"{li.pallet_id}-X{li.qty_in_stack}", ha="center", va="center", fontsize=7, weight="bold")
        legend_y = 0
        for pid, color in pallet_colors.items():
            ax.add_patch(Rectangle((c.l + 5, legend_y), 15, 15, facecolor=color, edgecolor="black"))
            ax.text(c.l + 25, legend_y + 7, pid, va="center", fontsize=9)
            legend_y += 20
        ax.set_aspect("equal")
        ax.set_xlim(0, c.l + 150)
        ax.set_ylim(0, max(c.w, legend_y))
        ax.set_title(f"2D Layout – {cid}", fontsize=12, weight="bold")
        plt.tight_layout()
        fname = os.path.join(folder, f"{cid}_2D.png")
        plt.savefig(fname, dpi=SAVE_DPI)
        plt.close()
        files.append('/' + fname.replace("\\","/"))  # relative path
    return files

# ------------------ HARD CODED CONTAINERS ------------------
containers_list = [
    {"id": "C1", "type": "40ft", "length": 1200, "width": 230, "height": 240},
    {"id": "C2", "type": "20ft", "length": 600,  "width": 230, "height": 240},
    {"id": "C3", "type": "40ft HC", "length": 1200, "width": 230, "height": 270},
]

# ------------------ ROUTES ------------------
@app.route("/")
def index():
    # Load previous selections from session
    selected_container_ids = session.get("selected_containers", [])
    pallets_data = session.get("pallets_data", [])
    return render_template("index.html", containers=containers_list, selected_container_ids=selected_container_ids, pallets_data=pallets_data)

@app.route("/run", methods=["POST"])
def run():
    selected_container_ids = request.form.getlist("container")
    if len(selected_container_ids) > 3:
        selected_container_ids = selected_container_ids[:3]

    # Store in session
    session["selected_containers"] = selected_container_ids

    # Extract pallets
    pallets = []
    pallets_data = []
    pallet_ids = request.form.getlist("pallet_id[]")
    lengths = request.form.getlist("length[]")
    widths = request.form.getlist("width[]")
    heights = request.form.getlist("height[]")
    quantities = request.form.getlist("quantity[]")
    for i in range(len(pallet_ids)):
        try:
            pid = pallet_ids[i]
            l = float(lengths[i])
            w = float(widths[i])
            h = float(heights[i])
            qty = int(quantities[i])
            pallets.append(PalletSpec(pid,l,w,h,qty))
            pallets_data.append({"pid":pid,"l":l,"w":w,"h":h,"qty":qty})
        except: continue

    # Save pallets in session
    session["pallets_data"] = pallets_data

    selected_containers = [
        ContainerSpec(
            id=c["id"], l=c["length"], w=c["width"], h=c["height"], qty=1, type=c["type"]
        ) for c in containers_list if c["id"] in selected_container_ids
    ]

    loaded, missed = pack(selected_containers, pallets)
    images = draw_2d_all(loaded, selected_containers, LAYOUT_IMAGE_DIR)

    return render_template("results.html", loaded=loaded, missed=missed, images=images)

@app.route("/clear", methods=["GET"])
def clear():
    session.pop("selected_containers", None)
    session.pop("pallets_data", None)
    return redirect(url_for("index"))

@app.route("/import_excel", methods=["POST"])
def import_excel():
    file = request.files["file"]
    if not file: return jsonify({"error":"No file uploaded"})
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    df = pd.read_excel(filepath, usecols="A:E", names=["ID","L","W","H","Qty"], skiprows=0)
    pallets=[]
    for _,r in df.iterrows():
        if pd.isna(r["ID"]): continue
        pallets.append({
            "pid":str(r["ID"]).strip(),
            "l":float(r["L"]),
            "w":float(r["W"]),
            "h":float(r["H"]),
            "qty":int(r["Qty"])
        })
    return jsonify(pallets)

if __name__ == "__main__":
    import webbrowser
    from threading import Timer
    def open_browser(): webbrowser.open_new("http://127.0.0.1:5001/")
    Timer(1, open_browser).start()
    app.run(debug=False, port=5001)

