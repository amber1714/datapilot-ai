import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"


class DataPilotAPIError(Exception):
    pass


def _handle_response(response):
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if response.ok:
        return payload

    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
    else:
        detail = response.text or "Unknown API error"

    raise DataPilotAPIError(
        f"API request failed ({response.status_code}): {detail}"
    )


def health_check(api_url=DEFAULT_API_URL):
    try:
        response = requests.get(
            f"{api_url}/health",
            timeout=10,
        )
        return _handle_response(response)
    except requests.RequestException as exc:
        raise DataPilotAPIError(
            f"Could not connect to FastAPI at {api_url}. "
            f"Make sure the backend is running. Details: {exc}"
        ) from exc


def analyze_dataset(
    file_name,
    file_bytes,
    api_url=DEFAULT_API_URL,
):
    try:
        response = requests.post(
            f"{api_url}/analyze",
            files={
                "file": (
                    file_name,
                    file_bytes,
                    "text/csv",
                )
            },
            timeout=60,
        )
        return _handle_response(response)
    except requests.RequestException as exc:
        raise DataPilotAPIError(
            f"Dataset analysis request failed: {exc}"
        ) from exc


def detect_problem(
    file_name,
    file_bytes,
    target_column,
    api_url=DEFAULT_API_URL,
):
    try:
        response = requests.post(
            f"{api_url}/detect-problem",
            data={
                "target_column": target_column,
            },
            files={
                "file": (
                    file_name,
                    file_bytes,
                    "text/csv",
                )
            },
            timeout=60,
        )
        return _handle_response(response)
    except requests.RequestException as exc:
        raise DataPilotAPIError(
            f"Problem-detection request failed: {exc}"
        ) from exc


def train_models(
    file_name,
    file_bytes,
    target_column,
    test_size,
    api_url=DEFAULT_API_URL,
):
    try:
        response = requests.post(
            f"{api_url}/train",
            data={
                "target_column": target_column,
                "test_size": str(test_size),
            },
            files={
                "file": (
                    file_name,
                    file_bytes,
                    "text/csv",
                )
            },
            timeout=180,
        )
        return _handle_response(response)
    except requests.RequestException as exc:
        raise DataPilotAPIError(
            f"Model-training request failed: {exc}"
        ) from exc
