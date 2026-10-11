"""Single- and multi-round execution using an injected model and recorder."""
from .parse import parse_stream

def run_rounds(context, model, work, recorder):
    records = []
    for number in range(1, context.rounds + 1):
        prompt = context.prompt(number)
        recorder.context(context)
        response = model.generate(prompt, context.arm, work, context.session_flags(number))
        solution_path = work / "solution.v"
        solution = solution_path.read_text() if solution_path.exists() else None
        recorder.response(number, response, solution)
        _, result, _, last, _ = parse_stream(response.stdout)
        error = None
        if response.timed_out:
            error = "model timeout"
        elif response.returncode != 0 or result.get("is_error") or result.get("api_error_status"):
            error = str(result.get("result") or response.stderr or "model process failed")
        elif not result:
            error = "missing model result event"
        context.messages.append({"round": number, "role": "assistant", "content": last})
        recorder.context(context)
        records.append(dict(round=number, response=response, solution=solution, error=error))
        if error:
            break
    return records
