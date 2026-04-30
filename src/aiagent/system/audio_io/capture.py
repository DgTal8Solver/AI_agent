import numpy as np
import queue
from threading import Thread, Event
import sounddevice as sd
from typing import Literal, Optional, Callable, Union

class AudioCapture:

    def __init__(
        self,
        samplerate: int,
        blocksize: int = 512,
        channels: int = 1,
        dtype: Literal['float32', 'float16', 'int16', 'int8'] = 'float32',
        device: Optional[Union[str, int]] = None,
        func: Optional[Callable[[np.ndarray], None]] = None
    ) -> None:
        r"""
        Args:
            func (callable): 
                User-supplied function to reproducing special audio processing 
                to an active stream.

                .. It must have this signature:
                    def func(data: numpy.ndarray) -> None:
                        ...
                        # Need to change original data
                        data = ...
        """

        self._samplerate = samplerate
        self._blocksize = blocksize
        self._channels = channels
        self._dtype = dtype
        self._device = device or sd.default.device
        self._max_duration = blocksize / samplerate * 10

        self._stream: Optional[sd.InputStream] = None
        self._user_func = func

        self._raw_queue = queue.Queue()
        self._processed_queue = queue.Queue()

        self._running = False

        self._process_thread: Optional[Thread] = None
        self._stop_event = Event()

    def _callback(
        self, 
        data: np.ndarray, 
        frames: int, 
        time, status
    ) -> None:
        if status:
            print(f"Audio status: {status}")

        chunk = data.copy()
        self._raw_queue.put(chunk)

    def _process_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._raw_queue.get(timeout = self._max_duration)
                if self._user_func:
                    self._user_func(chunk)
                self._processed_queue.put(chunk)
            except queue.Empty:
                pass

    def start(self) -> None:
        if self._running:
            raise RuntimeError("AudioCapture is already running!")

        self._stream = sd.InputStream(
            samplerate = self._samplerate,
            blocksize = self._blocksize,
            channels = self._channels,
            dtype = self._dtype,
            device = self._device,
            callback = self._callback
        )

        self._stop_event.clear()
        self._process_thread = Thread(target = self._process_loop, daemon = False)

        self._stream.start()
        self._process_thread.start()

        self._running = True
        print(f"AudioCapture started (blocksize = {self._blocksize})")
    
    def stop(self) -> None:
        if not self._running:
            return

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        if self._process_thread:
            self._stop_event.set()
            self._process_thread.join()
            self._process_thread = None

        self._running = False
        print("AudioCapture stopped")

    def get_chunk(self) -> Union[np.ndarray, None]:
        try:
            return self._processed_queue.get_nowait()
        except queue.Empty:
            return None