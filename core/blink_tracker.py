import logging
from dataclasses import dataclass
from numba_pokemon_prngs.xorshift import Xorshift128
from .rng_recovery import extract_known_bit_length_float, recover_rng_state_float


@dataclass
class Blink:
    timestamp: float
    duration: float
    since_last: float | None


class BlinkTracker:
    def __init__(self, on_completion_reciever=None):
        self.on_completion_reciever = on_completion_reciever
        # minimum time spent detecting/not detecting to warrant changing state
        self.epsilon = 0.1
        self.reset()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def reset(self):
        self.current_state = True
        self.last_confirmed = None
        self.before_last_confirmed = None
        self.last_blink = None

    def on_blink(self, blink: Blink):
        pass

    def process_data(self, data):
        (detected, timestamp) = data
        if self.last_confirmed is None:
            self.current_state = detected
            self.last_confirmed = timestamp
        elif detected == self.current_state:
            self.last_confirmed = timestamp
        else:
            delta = timestamp - self.last_confirmed
            if delta > self.epsilon:
                if detected and self.before_last_confirmed is not None:
                    length_of_blink = self.last_confirmed - self.before_last_confirmed
                    blink = Blink(
                        self.last_confirmed,
                        length_of_blink,
                        (
                            None
                            if self.last_blink is None
                            else timestamp - self.last_blink
                        ),
                    )
                    self.on_blink(blink)
                    self.last_blink = timestamp
                self.current_state = detected
                self.before_last_confirmed = self.last_confirmed
                self.last_confirmed = timestamp


class MunchlaxBlinkTracker(BlinkTracker):
    def __init__(self, on_completion_reciever=None, on_progress_reciever=None):
        # known extra padding or error in measurement subtracted from observed values
        self.offset = 0.287155
        # allowed +/- inaccuracy from true value
        self.leeway = 0.1
        self.on_progress_reciever = on_progress_reciever
        super().__init__(on_completion_reciever)

    def set_offset(self, offset):
        self.offset = offset

    def set_leeway(self, leeway):
        self.leeway = leeway

    def reset(self):
        self.entropy = 0
        self.blinks = []
        super().reset()

    def on_blink(self, blink: Blink):
        logging.info(" Munchlax Blink: %s", blink)
        if blink.since_last is None:
            return
        if not (3.0 <= blink.since_last - self.offset <= 12.0):
            logging.warning(
                " Blink out of range (%.6f)", blink.since_last - self.offset
            )
        self.blinks.append(blink)
        self.entropy += extract_known_bit_length_float(
            3.0, 12.0, self.leeway, blink.since_last - self.offset
        )
        logging.info(" Current entropy: %d/128", self.entropy)
        if self.on_progress_reciever is not None:
            self.on_progress_reciever.emit(self.entropy)
        if self.entropy >= 128:
            logging.info(
                " Estimated required entropy met (%d bits in %d observations)",
                self.entropy,
                len(self.blinks[1:]),
            )
            rng_state = recover_rng_state_float(
                3.0,
                12.0,
                self.leeway,
                [b.since_last - self.offset for b in self.blinks[1:]],
            )
            if rng_state is None:
                logging.warning(" Insufficient entropy, failed to recover rng state")
                return
            logging.info(" Recovered rng state (old): %08X %08X %08X %08X", *rng_state)

            test_rng = Xorshift128(*rng_state)
            total_offset = 0
            min_offset = float("inf")
            max_offset = float("-inf")
            for i, blink in enumerate(self.blinks[1:]):
                assert blink.since_last is not None
                real_value = test_rng.next_float_randrange(3.0, 12.0)
                observed_value = blink.since_last
                offset = observed_value - real_value
                total_offset += offset
                min_offset = min(min_offset, offset)
                max_offset = max(max_offset, offset)
                logging.info(" Blink %d:", i)
                logging.info(" \tReal: %.6f", real_value)
                logging.info(" \tObserved: %.6f", observed_value)
                logging.info(" \tOffset: %.6f", offset)
            average_offset = total_offset / len(self.blinks[1:])
            logging.info(" Average offset: %.6f", average_offset)
            logging.info(" Negative leeway: %.6f", average_offset - min_offset)
            logging.info(" Positive leeway: %.6f", max_offset - average_offset)

            if self.on_completion_reciever is not None:
                self.on_completion_reciever.emit(
                    (tuple(test_rng.state), self.blinks[-1])
                )

            test_rng.next()
            logging.info(
                " Recovered rng state (current): %08X %08X %08X %08X", *test_rng.state
            )
            self.reset()
