"""Parse model event streams independently of model execution."""
import json

def parse_stream(stdout):
    calls, result, peak, last_text, iters = {}, {}, 0, "", 0
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "assistant":
            msg = ev.get("message", {})
            u = msg.get("usage", {})
            peak = max(peak, u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                       + u.get("cache_creation_input_tokens", 0))
            for b in msg.get("content", []):
                if b.get("type") == "tool_use":
                    n = b["name"].replace("mcp__ic__", "")
                    calls[n] = calls.get(n, 0) + 1
                    if n in ("Write", "Edit") and "solution" in json.dumps(b.get("input", {}))[:300]:
                        iters += 1
                elif b.get("type") == "text":
                    last_text = b["text"]
        elif ev.get("type") == "result":
            result = ev
    return calls, result, peak, last_text, iters

def summarize(streams):
    calls, usage, cost, turns, peak, last, iterations = {}, {}, 0.0, 0, 0, "", 0
    for stream in streams:
        c, result, pk, text, count = parse_stream(stream)
        for name, value in c.items():
            calls[name] = calls.get(name, 0) + value
        for name, value in result.get("usage", {}).items():
            if isinstance(value, (int, float)):
                usage[name] = usage.get(name, 0) + value
        cost += result.get("total_cost_usd") or 0
        turns += result.get("num_turns") or 0
        peak, last, iterations = max(peak, pk), text or last, iterations + count
    return dict(tool_calls=calls, usage=usage, cost_usd=cost, turns=turns,
                peak_ctx_tokens=peak, final_message=last, iterations=iterations)
