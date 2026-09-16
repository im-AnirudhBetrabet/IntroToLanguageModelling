from collections import Counter

def explore_names():
    print("="*30)
    print(">> Beginning data exploration")
    with open(file="names.txt", mode='r') as f:
        data = f.read().splitlines()
    print(f">>> File has {len(data)} data points")
    print(f">>> Maximum characters in a word is {max(len(d) for d in data)}")
    print(f">>> Minimum characters in a word is {min(len(d) for d in data)}")
    print(f">>> Average characters in a word is {round(sum(len(d) for d in data) / len(data), 2)}")
    print(f">>> Empty names {sum(not d for d in data)}")

    chars = sorted(set("".join(data)))
    print(f">>> Unique characters in the dataset are {len(chars)}: ( {', '.join(chars)} )")

    char_counts = Counter("".join(data))

    print("\n>>> Character frequencies:")
    for char, count in char_counts.most_common():
        print(f"    {char}: {count}")

    print("\n>>> Sample names:")
    for name in data[:10]:
        print(f"    {name}")
    print("=" * 30)
if __name__ == "__main__":
    explore_names()