from typing import AsyncGenerator
import json
from importlib import metadata
from exorde_data.models import *
from madtypes import schema

from . import scraping


def install_modules():
    raise NotImplemented()


def get_scraping_module(url: str):
    for module_name in dir(scraping):
        if module_name in url:
            return getattr(scraping, module_name)
    return None


async def query(url: str) -> AsyncGenerator[Item, None]:
    scraping_module = get_scraping_module(url)
    if not scraping_module:
        raise NotImplemented(f"There is no scraping module for {url}")
    async for item in scraping_module.query(url):
        print("\n")
        print(item)
        print("\n")
        yield item


def print_schema():
    schem = schema(
        Analyzed,
        **{
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": f'https://github.com/exorde-labs/exorde/repo/tree/v{metadata.version("exorde_data")}/exorde/schema/schema.json',
        },
    )
    try:
        print(
            json.dumps(
                schem,
                indent=4,
            )
        )
    except Exception as err:
        print(err)
        print(schem)


__all__ = ["scraping"]
