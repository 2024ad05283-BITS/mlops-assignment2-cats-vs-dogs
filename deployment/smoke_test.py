"""
smoke_test.py

Production-ready smoke tests for deployment.

Checks
------
1. Health endpoint
2. Prediction endpoint

Exit Codes
----------
0 -> Success
1 -> Failure
"""

from pathlib import Path
import sys

import requests

API_URL = "http://localhost:8000"

HEALTH_URL = f"{API_URL}/health"

PREDICT_URL = f"{API_URL}/predict"

#TEST_IMAGE = Path("../data/processed/test/Cat").glob("*")
TEST_IMAGE = (Path(__file__).parent / "../data/processed/test/Cat").glob("*")

def get_test_image():

    images = list(TEST_IMAGE)

    if not images:

        raise FileNotFoundError(
            "No test image found."
        )

    return images[0]


def health_check():

    print("=" * 60)
    print("Health Check")
    print("=" * 60)

    response = requests.get(
        HEALTH_URL,
        timeout=30,
    )

    if response.status_code != 200:

        print("FAILED")

        print(response.text)

        sys.exit(1)

    print("PASSED")

    print(response.json())


def prediction_check():

    print("=" * 60)
    print("Prediction Check")
    print("=" * 60)

    image_path = get_test_image()

    with open(image_path, "rb") as image:

        response = requests.post(

            PREDICT_URL,

            files={

                "file": image,

            },

            timeout=30,

        )

    if response.status_code != 200:

        print("FAILED")

        print(response.text)

        sys.exit(1)

    result = response.json()

    print("PASSED")

    print(result)

    required_keys = (

        "label",

        "confidence",

        "probabilities",

    )

    for key in required_keys:

        if key not in result:

            print(f"Missing key : {key}")

            sys.exit(1)


def main():

    print("=" * 60)
    print("Running Smoke Tests")
    print("=" * 60)

    health_check()

    prediction_check()

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":

    try:

        main()

        sys.exit(0)

    except Exception as e:

        print("=" * 60)

        print("SMOKE TEST FAILED")

        print("=" * 60)

        print(e)

        sys.exit(1)