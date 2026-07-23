import argparse
import re
from pathlib import Path

import pandas as pd


def parse_momp_file(file_content):
    lines = file_content.strip().split("\n")
    results = []

    i = 0
    while i < len(lines):
        if re.match(r"T is length \d+, and m is set to 512", lines[i].strip()):
            length = re.search(r"T is length (\d+),", lines[i].strip()).group(1)
            i += 1

            while i < len(lines) and not lines[i].strip().startswith("====== MOMP ======"):
                i += 1
            i += 1

            while i < len(lines) and not (
                    lines[i].strip().startswith("MOMP : Tpaa1in1 |")):
                i += 1

            if i < len(lines):
                line = lines[i].strip()
                curly_match = re.search(r"\{(\d+,\s*\d+)\}", line)
                time_match = re.search(r"Time:\s*([\d.]+)", line)

                curly = curly_match.group(1) if curly_match else None
                time = time_match.group(1) if time_match else None

                if curly and time:
                    results.append({
                        "length": int(length),
                        "curly": curly,
                        "time": float(time),
                    })
        i += 1

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MOMP motif locations and runtimes from a text log."
    )
    parser.add_argument("input", type=Path, help="MOMP text log to parse.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path. Prints rows when omitted.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = parse_momp_file(args.input.read_text())
    frame = pd.DataFrame(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")
    else:
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
