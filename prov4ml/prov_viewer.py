import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog
import argparse
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import pandas as pd
import time
import json
import seaborn as sns

TKWINDOW = None
SIDEFRAME = None
SIDEPANEL = None
CANVAS = None
CANVAS_WIDGET = None
SELECTED_METRIC = None
CURRENT_FILE_PATH = None
CURRENT_DATA = None
NEEDS_REFRESH = None

def get_metric(data, metric, time_in_sec=False, time_incremental=False, sort_by=None, start_at=None, end_at=None):

    if metric not in data["entity"].keys(): 
        raise AttributeError

    try:
        epochs = eval(data["entity"][metric]["prov-ml:metric_epoch_list"])
        values = eval(data["entity"][metric]["prov-ml:metric_value_list"])
        times = eval(data["entity"][metric]["prov-ml:metric_timestamp_list"])

        start_at = 0 if start_at is None else start_at
        end_at = len(epochs) if end_at is None else end_at        

        epochs = epochs[start_at:end_at]
        values = values[start_at:end_at]
        times = times[start_at:end_at]
    except: 
        return pd.DataFrame(columns=["epoch", "value", "time"])
    
    df = pd.DataFrame({"epoch": epochs, "value": values, "time": times})#.drop_duplicates()
    if time_incremental: 
        df["time"] = df["time"].diff().fillna(0)

    if sort_by is not None: 
        df = df.sort_values(by=sort_by)
    
    return df

def get_metrics(data, keyword=None):
    ms = data["entity"].keys()
    if keyword is None:
        return ms
    else:
        return [m for m in ms if keyword in m]

def get_file_data(file_path): 
    time.sleep(1)
    data = json.load(open(file_path))
    metrics = get_metrics(data, "TRAINING")
    metrics.remove("Indices_Context.TRAINING")
    return data, metrics

def plot_data(): 
    global CANVAS, CANVAS_WIDGET, TKWINDOW, CURRENT_DATA, SELECTED_METRIC, NEEDS_REFRESH

    if NEEDS_REFRESH: 
        NEEDS_REFRESH.set(False)
    else: 
        return; 

    metric_data = get_metric(CURRENT_DATA, SELECTED_METRIC)
    indices = get_metric(CURRENT_DATA, metric="Indices_Context.TRAINING", sort_by="time")["value"]

    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    ax.clear()
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=metric_data, x="time", y="value", color='tab:blue', ax=ax)
    ax.set_ylabel(f"{SELECTED_METRIC}")
    ax.set_xlabel("Batch")
    ax.set_xticks(metric_data["time"])
    ax.set_xticklabels(indices, rotation=90)

    colors = []
    for v in indices:
        if "100" in v: 
            colors.append('tab:red')
            continue
        elif "101" in v: 
            colors.append('tab:red')
            continue
        elif "102" in v: 
            colors.append('tab:red')
            continue
        elif "103" in v: 
            colors.append('tab:red')
            continue
        else: 
            colors.append('black')

    tick_labels = ax.get_xticklabels()
    for label, color in zip(tick_labels, colors):
        label.set_color(color)

    epochs = []
    for e in range(1, len(metric_data["epoch"])): 
        if metric_data["epoch"][e-1] != metric_data["epoch"][e]: 
            epochs.append(metric_data["time"][e-1])
    for epoch in epochs:  # `epochs` should be a list or array of x-values corresponding to epochs
        ax.axvline(x=epoch, color='red', linestyle='--', linewidth=1)

    if CANVAS_WIDGET is not None:
        CANVAS_WIDGET.destroy()
    CANVAS = FigureCanvasTkAgg(fig, master=TKWINDOW)
    CANVAS_WIDGET = CANVAS.get_tk_widget()
    CANVAS_WIDGET.pack(side=tk.LEFT, fill=tk.Y, expand=1)

def change_metric(m): 
    global SELECTED_METRIC
    if SELECTED_METRIC != m: 
        SELECTED_METRIC = m
        plot_data()

def clear_side_panel(): 
    global SIDEPANEL, SIDEFRAME
    if SIDEPANEL is not None: 
        SIDEPANEL.destroy()
    SIDEPANEL = tk.Listbox(SIDEFRAME)
    SIDEPANEL.pack(fill='both', expand=True, pady=10)

def update_side_panel(metrics): 
    global SIDEPANEL
    btns = [tk.Button(SIDEPANEL, text=f"{m}", 
                  command=lambda metric=m: change_metric(metric),
                   activebackground="blue", 
                   activeforeground="white",
                   anchor="center",
                   bd=3,
                   bg="lightgray",
                   cursor="hand2",
                   disabledforeground="gray",
                   fg="black",
                   font=("Arial", 12),
                   height=2,
                   highlightbackground="black",
                   highlightcolor="green",
                   highlightthickness=2,
                   justify="center",
                   overrelief="raised",
                   padx=10,
                   pady=5,
                   width=15,
                   wraplength=100) for m in metrics]

    [button.pack() for button in btns]

class DirectoryHandler(FileSystemEventHandler):
    def on_created(self, event):
        global CURRENT_FILE_PATH, SELECTED_METRIC, CURRENT_DATA, NEEDS_REFRESH

        if not event.is_directory:
            file_path = event.src_path
            if str(file_path).endswith(".json"): 
                try:
                    print(f"New file detected: {file_path}")
                    # self.queue.put(file_path)  # Send updates to the main thread

                    if CURRENT_FILE_PATH != file_path: 
                        CURRENT_FILE_PATH = file_path
                        CURRENT_DATA, metrics = get_file_data(CURRENT_FILE_PATH)                    
                    if SELECTED_METRIC is None: 
                        SELECTED_METRIC = metrics[0]

                    clear_side_panel()
                    update_side_panel(metrics)

                    NEEDS_REFRESH.set(True)

                except Exception as e:
                    print(f"Error processing directory {file_path}: {e}")

# Watchdog observer thread
def observer_thread():
    global WATCH_DIR

    event_handler = DirectoryHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=True)
    observer.start()

    print(f"Monitoring directory for new subdirectories: {WATCH_DIR}")
    try:
        while True:
            time.sleep(1)  # Keep thread running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    plt.close()

def save_to_pdf(): 
    global CANVAS 

    if CANVAS is None: 
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )
    if file_path:
        CANVAS.figure.savefig(file_path, bbox_inches='tight')
        print(f"Chart saved as {file_path}")

def main(watch_dir): 
    global TKWINDOW, SIDEFRAME, CANVAS_WIDGET, CANVAS, WATCH_DIR, NEEDS_REFRESH

    WATCH_DIR = watch_dir

    observer = Thread(target=observer_thread, daemon=True)
    observer.start()

    TKWINDOW = tk.Tk()
    TKWINDOW.title("Prov Viewer")
    SIDEFRAME = tk.Frame(TKWINDOW)  
    SIDEFRAME.pack(side='right', fill='y') 
    BUTTON = tk.Button(
        SIDEFRAME, 
        command=save_to_pdf, 
        text="Save to pdf", 
        bg="lightblue",       # Background color
        fg="darkblue",        # Text color
        font=("Helvetica", 12, "bold"),  # Font style
        relief="raised",      # Button relief style
        bd=2, 
        width=15,
        wraplength=100)
    BUTTON.pack(fill='x', pady=10) 

    NEEDS_REFRESH = tk.BooleanVar(master=TKWINDOW, value=False)
    NEEDS_REFRESH.trace_add("write", lambda a, b, c: plot_data())

    TKWINDOW.mainloop()

    plt.close()
    if CANVAS_WIDGET is not None: 
        CANVAS_WIDGET.destroy()
    del CANVAS_WIDGET
    del CANVAS
    TKWINDOW.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor a directory and update a plot when new subdirectories are added.")
    parser.add_argument("--dir",type=str,help="Path to the directory to monitor.")
    args = parser.parse_args()

    # Check if the directory exists
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)

    main(args.dir)
