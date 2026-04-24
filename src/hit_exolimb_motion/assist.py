from __future__ import annotations


def plan_support_events(
    overhead_segments: list[dict[str, object]],
    support_lead_time_s: float = 0.2,
    release_delay_s: float = 0.3,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for index, segment in enumerate(overhead_segments):
        start = float(segment["start_time"])
        end = float(segment["end_time"])
        side = str(segment.get("side", "auto"))
        events.extend(
            [
                {
                    "event": "prepare_support",
                    "segment_index": index,
                    "time": max(0.0, start - support_lead_time_s),
                    "target": "overhead_tool_or_workpiece",
                    "side": side,
                },
                {
                    "event": "hold_support",
                    "segment_index": index,
                    "time": start,
                    "target": "overhead_tool_or_workpiece",
                    "side": side,
                },
                {
                    "event": "release_support",
                    "segment_index": index,
                    "time": end + release_delay_s,
                    "target": "overhead_tool_or_workpiece",
                    "side": side,
                },
            ]
        )
    return events

