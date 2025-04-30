import time
import os
import subprocess
from threading import Lock, Thread

import cv2
import gradio as gr
import numpy as np
import torch
from tqdm import tqdm

from remover import inpaint, load_image_mask, load_model, pre_process, to_bhwc

model_path = "big-lama.pt"
if not os.path.exists(model_path):
    print("Downloading model...")
    subprocess.run(
        [
            "wget",
            "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
            "-O",
            model_path,
        ]
    )

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

MODEL = load_model(model_path, device=device)


class Listener:
    task_queue = []
    lock = Lock()
    thread = None

    @classmethod
    def _process_tasks(cls):
        while True:
            task = None
            with cls.lock:
                if cls.task_queue:
                    task = cls.task_queue.pop(0)

            if task is None:
                time.sleep(0.001)
                continue

            func, args, kwargs = task
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"Error in listener thread: {e}")

    @classmethod
    def add_task(cls, func, *args, **kwargs):
        with cls.lock:
            cls.task_queue.append((func, args, kwargs))

        if cls.thread is None:
            cls.thread = Thread(target=cls._process_tasks, daemon=True)
            cls.thread.start()


def async_run(func, *args, **kwargs):
    Listener.add_task(func, *args, **kwargs)


class FIFOQueue:
    def __init__(self):
        self.queue = []
        self.lock = Lock()

    def push(self, item):
        with self.lock:
            self.queue.append(item)

    def pop(self):
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
            return None

    def top(self):
        with self.lock:
            if self.queue:
                return self.queue[0]
            return None

    def next(self):
        while True:
            with self.lock:
                if self.queue:
                    return self.queue.pop(0)

            time.sleep(0.001)


class AsyncStream:
    def __init__(self):
        self.input_queue = FIFOQueue()
        self.output_queue = FIFOQueue()


class ProcessManager:
    def __init__(self):
        self.mask = None
        self.frames = None
        self.fps = None

    def load_video(self, video):
        cap = cv2.VideoCapture(video)
        # convert video to frames
        self.frames = []
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(frame)
        cap.release()

        return self.frames[0]

    def process(
        self,
        input,
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ):
        mask = input["layers"][0][:, :, -1, None]
        image = input["background"][..., :-1]  # HWC

        if not self.frames:
            self.frames = [image]
        # Process video frames
        stream = AsyncStream()

        def worker(frames, mask):
            processed_frames = []
            for i in tqdm(range(len(frames)), desc="Processing..."):
                frame, frame_mask = load_image_mask(frames[i], mask)
                frame, frame_mask = pre_process(frame, frame_mask, device=device)
                results = inpaint(MODEL, frame, frame_mask, seed=0)
                results = to_bhwc(results)
                result_frame = (results[0].detach().cpu().numpy() * 255).astype(
                    np.uint8
                )
                if i == 0:
                    stream.output_queue.push(
                        (
                            "start",
                            (frames[i], result_frame),
                        )
                    )
                processed_frames.append(result_frame)

            # convert to video
            if len(frames) > 1:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                height, width, _ = processed_frames[0].shape
                out = cv2.VideoWriter("output.mp4", fourcc, self.fps, (width, height))
                for frame in processed_frames:
                    frame = frame[..., ::-1]  # Convert RGB to BGR
                    out.write(frame)
                out.release()
                # save video
                stream.output_queue.push(("end", "output.mp4"))
            else:
                stream.output_queue.push(("end", None))

        async_run(worker, self.frames, mask)

        while True:
            flag, data = stream.output_queue.next()
            if flag == "start":
                start_frame = data
                yield (start_frame, None, gr.update(interactive=False))
            if flag == "end":
                output_filename = data
                yield (start_frame, output_filename, gr.update(interactive=True))
                break


manager = ProcessManager()

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row(equal_height=True):
            image_input = gr.ImageMask(label="Image", sources="upload")
            image_result = gr.ImageSlider(label="Processed Image")
        with gr.Row(equal_height=True):
            video_input = gr.Video(label="Video", sources="upload")
            video_result = gr.Video(label="Processed Video")

    process_btn = gr.Button("Process")

    # Define the event listeners
    process_btn.click(
        fn=manager.process,
        inputs=[image_input],
        outputs=[image_result, video_result, process_btn],
    )
    video_input.upload(
        fn=manager.load_video, inputs=[video_input], outputs=[image_input]
    )
    video_input.clear(
        fn=lambda: (None, None, None),
        inputs=[],
        outputs=[image_input, image_result, video_result],
    )
    image_input.clear(fn=lambda: None, inputs=[], outputs=[image_result])


# Launch the app
demo.launch(show_error=True)
