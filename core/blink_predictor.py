from PySide6.QtCore import QThread
from numba_pokemon_prngs.xorshift import Xorshift128
from time import perf_counter


class MunchlaxBlinkPredictor(QThread):
    def __init__(self, on_blink_reciever, rng_state, last_blink, offset=0.285):
        super().__init__()
        self.running = True
        self.on_blink_reciever = on_blink_reciever
        self.rng = Xorshift128(*rng_state)
        self.advance = 0
        self.offset = offset
        self.next_blink = (
            last_blink + self.rng.next_float_randrange(3.0, 12.0) + self.offset
        )
        self.on_blink_reciever.emit((self.next_blink, perf_counter(), self.advance))

    def run(self):
        while self.running:
            current_time = perf_counter()
            while current_time > self.next_blink:
                self.next_blink += (
                    self.rng.next_float_randrange(3.0, 12.0) + self.offset
                )
                self.advance += 1
                self.on_blink_reciever.emit(
                    (self.next_blink, current_time, self.advance)
                )
            self.msleep(1)
