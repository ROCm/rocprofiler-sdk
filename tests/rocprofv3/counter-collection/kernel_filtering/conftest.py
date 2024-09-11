#!/usr/bin/env python3

import json
import pytest
import pandas as pd

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):

    parser.addoption(
        "--input-json-pass1",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-csv-pass1",
        action="store",
        help="Path to CSV file.",
    )

    parser.addoption(
        "--input-json-pass2",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-csv-pass2",
        action="store",
        help="Path to CSV file.",
    )

    parser.addoption(
        "--input-json-pass3",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-csv-pass3",
        action="store",
        help="Path to CSV file.",
    )

    parser.addoption(
        "--input-json-pass4",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-csv-pass4",
        action="store",
        help="Path to CSV file.",
    )

    parser.addoption(
        "--input-csv-pmc1",
        action="store",
        help="Path to CSV file.",
    )


@pytest.fixture
def input_csv_pass1(request):
    filename = request.config.getoption("--input-csv-pass1")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_csv_pass2(request):
    filename = request.config.getoption("--input-csv-pass2")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_csv_pass3(request):
    filename = request.config.getoption("--input-csv-pass3")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_csv_pass4(request):
    filename = request.config.getoption("--input-csv-pass4")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_csv_pmc1(request):
    filename = request.config.getoption("--input-csv-pmc1")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def input_json_pass1(request):
    filename = request.config.getoption("--input-json-pass1")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def input_json_pass2(request):
    filename = request.config.getoption("--input-json-pass2")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def input_json_pass3(request):
    filename = request.config.getoption("--input-json-pass3")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def input_json_pass4(request):
    filename = request.config.getoption("--input-json-pass4")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))
