#!/usr/bin/env python3

import csv
import pytest
import pandas as pd


def pytest_addoption(parser):
    parser.addoption("--basic-metrics-input", action="store", help="Path to csv file.")
    parser.addoption("--derived-metrics-input", action="store", help="Path to csv file.")


@pytest.fixture
def derived_metrics_input_data(request):
    filename = request.config.getoption("--derived-metrics-input")
    data = []
    if filename:
        with open(filename, "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

        return data


@pytest.fixture
def basic_metrics_input_data(request):
    filename = request.config.getoption("--basic-metrics-input")
    data = []
    if filename:
        with open(filename, "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)

        return data
