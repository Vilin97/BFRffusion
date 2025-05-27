#%%
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os

base_logdir = "/gscratch/krishna/vilin/BFRffusion/experiments"
logdirs = glob.glob(f"{base_logdir}/*/tensorboard/version_*")

for logdir in logdirs:
    event_files = glob.glob(f"{logdir}/events.out*")
    if not event_files:
        print(f"No event files found in {logdir}.")
        continue
    for ef in event_files:
        print(f"\nReading: {ef}")
        ea = EventAccumulator(ef)
        ea.Reload()
        tags = ea.Tags()
        print("Scalars:", tags.get('scalars', []))
        print("Images:", tags.get('images', []))
        print("Histograms:", tags.get('histograms', []))
        print("Tensors:", tags.get('tensors', []))
        print("Text:", tags.get('text', []))
        print("Graphs:", tags.get('graph', []))
