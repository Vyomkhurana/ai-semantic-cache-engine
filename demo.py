import requests
import json
import time
import sys

BASE = "http://localhost:8000"


def pretty(d):
    print(json.dumps(d, indent=2))


def check_server():
    try:
        requests.get(f"{BASE}/cache/stats", timeout=3)
    except Exception:
        print("server not running. start it with:")
        print("  uvicorn main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)


def section(title):
    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)


def main():
    check_server()

    # clear any existing state
    requests.delete(f"{BASE}/cache")

    section("1. fresh cache - first query (miss expected)")
    r = requests.post(f"{BASE}/query", json={"query": "how does encryption work"})
    pretty(r.json())
    time.sleep(0.5)

    section("2. exact same query again (hit expected)")
    r = requests.post(f"{BASE}/query", json={"query": "how does encryption work"})
    pretty(r.json())
    time.sleep(0.5)

    section("3. paraphrase - different words, same meaning (hit expected)")
    r = requests.post(f"{BASE}/query", json={"query": "explain how cryptography works"})
    pretty(r.json())
    time.sleep(0.5)

    section("4. unrelated query (miss expected, new entry stored)")
    r = requests.post(f"{BASE}/query", json={"query": "nasa space shuttle program history"})
    pretty(r.json())
    time.sleep(0.5)

    section("5. paraphrase of unrelated (hit expected)")
    r = requests.post(f"{BASE}/query", json={"query": "history of the space shuttle"})
    pretty(r.json())
    time.sleep(0.5)

    section("6. cache stats")
    r = requests.get(f"{BASE}/cache/stats")
    pretty(r.json())

    section("7. flush and verify reset")
    requests.delete(f"{BASE}/cache")
    r = requests.get(f"{BASE}/cache/stats")
    pretty(r.json())

    print("\ndone.")


if __name__ == "__main__":
    main()
