import sys

from app.core.auth import key_store


def main():
    if len(sys.argv) < 2:
        print("Istemal: python -m scripts.create_key <naam> [rate_limit_per_min]")
        sys.exit(1)

    name = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    key = key_store.create(name, limit)
    print()
    print(f"Key '{name}' ban gayi (limit: {limit} requests/minute)")
    print()
    print(f"    {key}")
    print()
    print("Isay abhi copy kar lo. Yeh dobara nahi dikhegi (file mein sirf hash hai).")


if __name__ == "__main__":
    main()