"""Test all 8 chat suggestion questions through the Mistral-powered streaming API.

Sends each question to the /api/v1/chat/message/stream SSE endpoint
and checks that visualizations are produced.

Usage: python test_mistral_all_questions.py [--host cn0070] [--port 8000]
"""

import argparse
import json
import sys
import time

import requests

QUESTIONS = [
    "What is the activity of IFNG across different immune cell types in CIMA?",
    "Compare TNF activity between CIMA and Inflammation Atlas",
    "What cytokines are most differentially active in COVID-19 patients?",
    "How does IL-17A activity correlate with age in CD4 T cells?",
    "What are the top secreted proteins in tumor-associated macrophages?",
    "Explain what CytoSig activity scores mean",
    "Show me the validation metrics for the Inflammation Atlas",
    "Which organs show the highest IL-6 activity in scAtlas?",
]


def test_question(host: str, port: int, question: str, timeout: int = 180) -> dict:
    """Send a question to the streaming endpoint and parse SSE results."""
    url = f"http://{host}:{port}/api/v1/chat/message/stream"
    payload = {"content": question}

    result = {
        "question": question,
        "text_chunks": [],
        "tool_calls": [],
        "tool_results": [],
        "visualizations": [],
        "errors": [],
        "done": False,
    }

    try:
        resp = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=timeout,
            headers={"Accept": "text/event-stream"},
        )
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]  # strip "data: "
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            chunk_type = chunk.get("type", "")
            if chunk_type == "text":
                result["text_chunks"].append(chunk.get("content", ""))
            elif chunk_type == "tool_call":
                tc = chunk.get("tool_call", {})
                result["tool_calls"].append(
                    f"{tc.get('name', '?')}({json.dumps(tc.get('input', tc.get('arguments', {})), separators=(',', ':'))[:100]})"
                )
            elif chunk_type == "tool_result":
                tr = chunk.get("tool_result", {})
                result["tool_results"].append(tr.get("name", "?"))
            elif chunk_type == "visualization":
                viz = chunk.get("visualization", chunk.get("config", {}))
                result["visualizations"].append({
                    "type": viz.get("type", viz.get("viz_type", "?")),
                    "title": viz.get("title", "?"),
                })
            elif chunk_type == "done":
                result["done"] = True
            elif chunk_type == "error":
                result["errors"].append(chunk.get("content", "unknown error"))

    except requests.exceptions.Timeout:
        result["errors"].append("TIMEOUT")
    except Exception as e:
        result["errors"].append(str(e))

    result["full_text"] = "".join(result["text_chunks"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="cn0070")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--questions", type=str, default="all",
                        help="Comma-separated question numbers (1-8) or 'all'")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if args.questions == "all":
        indices = list(range(len(QUESTIONS)))
    else:
        indices = [int(x) - 1 for x in args.questions.split(",")]

    print("=" * 80)
    print(f"  Testing {len(indices)} questions through Mistral API at {args.host}:{args.port}")
    print("=" * 80)

    results = []
    for idx in indices:
        q = QUESTIONS[idx]
        qnum = idx + 1
        print(f"\n{'=' * 80}")
        print(f"  Q{qnum}: {q}")
        print(f"{'=' * 80}")

        t0 = time.time()
        result = test_question(args.host, args.port, q, timeout=args.timeout)
        elapsed = time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 1)
        results.append(result)

        has_viz = len(result["visualizations"]) > 0
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Tool calls ({len(result['tool_calls'])}):")
        for tc in result["tool_calls"]:
            print(f"    - {tc}")
        print(f"  Visualizations: {len(result['visualizations'])} {'✓' if has_viz else '✗ MISSING'}")
        for v in result["visualizations"]:
            print(f"    - {v['type']}: {v['title']}")
        if result["errors"]:
            print(f"  ERRORS: {result['errors']}")
        print(f"  Response preview: {result['full_text'][:200]}...")
        print(f"  Done: {result['done']}")

    # Summary
    print(f"\n\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    for i, r in enumerate(results):
        idx = indices[i]
        qnum = idx + 1
        n_viz = len(r["visualizations"])
        n_tc = len(r["tool_calls"])
        status = "✓" if n_viz > 0 or qnum == 6 else "✗"
        tc_names = r["tool_calls"]
        print(f"  Q{qnum}: {status} | {r['elapsed_seconds']}s | {n_tc} tools | {n_viz} viz | errors: {r['errors'] or 'none'}")

    # Save
    output_path = "/data/parks34/projects/2cytoatlas/logs/mistral_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
