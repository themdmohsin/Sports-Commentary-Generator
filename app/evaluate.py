"""Compare matched vs mismatched calls for unseen players using raw LM scores."""
from .data import examples, HELD_OUT, prompt
from .generator import Generator


def main():
    model = Generator()
    rows = list(examples(HELD_OUT))
    counts = [0, 0]
    for i, (_, event, actual) in enumerate(rows):
        # Rotate within the same player's six plays so names cannot solve it.
        other = rows[i // 6 * 6 + (i + 1) % 6][2]
        prefix = model.base.encode(prompt(event))
        for j, adapted in enumerate((False, True)):
            counts[j] += model.score(prefix, actual, adapted) > model.score(prefix, other, adapted)
        print(event, "=>", model.generate(event))
    print(f"Matched call preferred: base {counts[0]}/{len(rows)}, adapted {counts[1]}/{len(rows)}")


if __name__ == "__main__":
    main()

