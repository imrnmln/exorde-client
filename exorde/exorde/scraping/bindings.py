from exorde_data import query

from madframe.bindings import alias, wire
from madframe.routines import routine, infinite_generator
from exorde.scraping import generate_url

broadcast_formated_when, on_formated_data_do = wire(perpetual=True)

alias("url")(generate_url)
scrap = infinite_generator(lambda: True)(query)
routine(0.2, perpetuate=False, condition=lambda processing: not processing)(
    broadcast_formated_when(scrap)
)


__all__ = ["broadcast_formated_when", "on_formated_data_do"]
