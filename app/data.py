"""Original simulated fixtures. Each row specifies an action and two calls."""

ACTIONS = [
    ("goal", "scores a goal", "Goal for {p}!", "{p} finds the net!"),
    ("save", "saves a penalty", "{p} keeps the penalty out!", "Denied! {p} saves the spot kick."),
    ("three", "hits a three-pointer", "Three points for {p}!", "{p} drains it from beyond the arc!"),
    ("block", "blocks a shot", "{p} rejects the shot!", "A clean block from {p}!"),
    ("six", "hits a six", "{p} sends it over the rope for six!", "Six runs! {p} clears the boundary."),
    ("wicket", "takes a wicket", "A wicket for {p}!", "{p} breaks the partnership!"),
]
NAMES = ["Asha", "Mira", "Leo", "Noor", "Sam", "Ravi", "Alex", "Isha"]
HELD_OUT = ["Tara", "Dev"]


def examples(names=NAMES):
    for i, name in enumerate(names):
        for j, (kind, action, first, second) in enumerate(ACTIONS):
            yield kind, f"{name} {action}.", (first, second)[(i + j) % 2].format(p=name)


def prompt(event):
    return f"Event: {event.strip()}\nCommentary:"

