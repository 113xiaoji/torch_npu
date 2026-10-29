"""Emit the native R26 guard before any wrapper body operation."""
import json


def wrap_entry_body(schema, route, return_type, arguments, body):
    suffix = ', ' + ', '.join(arguments) if arguments else ''
    return (f'return c10_npu::eager_parallel::Invoke({json.dumps(schema)}, {json.dumps(route)}, '
            f'[&]() -> {return_type} {{\n{body}\n}}{suffix});')
