from logging import Formatter, getLogger, StreamHandler, DEBUG

L = getLogger(__name__)
formatter = Formatter("%(asctime)-15s - %(levelname)-8s - %(message)s")
handler = StreamHandler()
handler.setLevel(DEBUG)
handler.setFormatter(formatter)
L.setLevel(DEBUG)
L.addHandler(handler)
L.propagate = False
