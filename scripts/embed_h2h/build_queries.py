# flake8: noqa: E501
"""Regenerate queries.jsonl (hand-written labelled queries) and validate every label
against corpus.jsonl. Run from this directory after build_corpus.py."""

import json

NL = [
    (
        "Compute cosine similarity between two vectors, returning zero when either vector has near-zero norm",
        [("L:assess.py:_cosine_similarity", 2)],
    ),
    (
        "Detect where throughput stops growing as concurrency increases and pick that concurrency as the capacity knee",
        [("L:assess.py:_find_knee", 2), ("L:assess.py:auto_ramp_concurrency", 1)],
    ),
    (
        "Compute the relative gain between a previous and current measurement, guarding against a zero baseline",
        [("L:assess.py:_relative_gain", 2)],
    ),
    (
        "Measure time to first token by sending a request limited to one generated token",
        [("L:assess.py:measure_prefill_ttft", 2)],
    ),
    ("Compute a percentile from an already sorted list of numbers", [("L:assess.py:_pct", 2)]),
    (
        "Fire many chat requests in parallel and report throughput and latency statistics",
        [("L:assess.py:run_concurrent", 2)],
    ),
    (
        "Probe whether the model supports OpenAI-style tool calling without aborting the run if it fails",
        [("L:assess.py:_tool_probe", 2), ("L:assess.py:probe_tool_calls", 1)],
    ),
    (
        "Raise a clear error when the gateway rejects the API key with HTTP 401",
        [("L:assess.py:_unauthorized_error", 2)],
    ),
    (
        "Fetch the served model id and its maximum context length from the models endpoint",
        [("L:assess.py:served_model", 2)],
    ),
    (
        "Resolve the inbound gateway API key from environment variables with a fallback name",
        [("L:gateway/_config.py:_gateway_api_key", 2)],
    ),
    (
        "Parse a comma-separated list of alias=target pairs into a dictionary, skipping malformed entries",
        [("L:gateway/_config.py:_parse_aliases", 2)],
    ),
    (
        "Decide whether a backend is feasible based on an environment flag holding a falsy token",
        [("L:gateway/_config.py:_is_feasible", 2)],
    ),
    (
        "Parse the comma-separated mesh seed list, trimming whitespace and trailing slashes",
        [("L:gateway/_mesh_config.py:_parse_seeds", 2)],
    ),
    (
        "Parse an environment variable as a positive integer and raise a specific error for zero, negative or non-numeric values",
        [
            ("L:gateway/_mesh_config.py:_parse_positive_int", 2),
            ("L:gateway/_mesh_config.py:build_mesh_config", 1),
        ],
    ),
    (
        "Read the HTTP request body using the Content-Length header without crashing on bad values",
        [("L:gateway/_mesh_routes.py:_read_request_body", 2)],
    ),
    (
        "Parse a JSON body and return an empty dict when it is malformed",
        [("L:gateway/_mesh_routes.py:_parse_json_object", 2)],
    ),
    (
        "Reject a mesh member announcement whose schema version has a different major number",
        [("L:gateway/_mesh_wire.py:_check_schema_major", 2), ("L:gateway/_mesh_wire.py:decode", 1)],
    ),
    (
        "Background thread that periodically announces this member to seed nodes and roster members",
        [
            ("L:gateway/_mesh_routes.py:_heartbeat_loop", 2),
            ("L:gateway/_mesh_routes.py:_run_heartbeat_pass", 1),
        ],
    ),
    (
        "Resolve a requested model name like cortex-thor to the lane served by a specific mesh member",
        [
            ("L:gateway/_mesh_routing.py:find_member_lane", 2),
            ("L:gateway/_mesh_routing.py:find_suffixed_lane", 1),
        ],
    ),
    (
        "Compare two serving fingerprints field by field to decide whether two replicas are identical",
        [
            ("L:gateway/_mesh_routing.py:fingerprints_identical", 2),
            ("L:gateway/_mesh_routing.py:verify_member_roles", 1),
        ],
    ),
    (
        "Decide how to handle a WebSocket upgrade request for the realtime endpoint",
        [
            ("L:gateway/_realtime.py:plan_realtime_upgrade", 2),
            ("L:gateway/_realtime.py:is_websocket_upgrade", 1),
        ],
    ),
    (
        "Warn when two backends claim the same served model id",
        [
            ("L:gateway/_config.py:_warn_on_served_name_collisions", 2),
            ("L:gateway/_config.py:_claimed_model_ids", 1),
        ],
    ),
    (
        "Parse the LoRA adapter names declared for the hand backend",
        [
            ("L:gateway/_config.py:_hand_adapter_names", 2),
            ("L:gateway/_config.py:_hand_adapter_aliases", 1),
        ],
    ),
    (
        "Probe a backend's readiness endpoint and map the status code to ready, not ready or unknown",
        [("L:gateway/_readiness.py:probe_backend_ready", 2)],
    ),
    (
        "Convert shell-literal backslash-n and backslash-t sequences in message text into real newline and tab characters",
        [("C:cli/channel.py:_interpret_escapes", 2)],
    ),
    (
        "Map a SystemExit code that may be None, an int or a string to a return code and message",
        [("C:cli/_passthrough.py:_translate_exit", 2)],
    ),
    (
        "Reject bot names that are not a safe single path segment",
        [("C:cli/bot.py:_validate_name_or_raise", 2)],
    ),
    (
        "Poll a TCP port until it accepts connections, checking the PID file to ensure it is a culture server",
        [("C:cli/mesh.py:_wait_for_server_port", 2)],
    ),
    (
        "Parse key=value lines from systemctl show output into a dictionary",
        [
            ("C:doctor/checks.py:_parse_systemctl_show", 2),
            ("C:doctor/checks.py:_systemctl_show", 1),
        ],
    ),
    (
        "Unescape IRCv3 message tag values where unknown escapes drop the backslash",
        [
            ("C:protocol/message.py:_unescape_tag_value", 2),
            ("C:protocol/message.py:Message._parse_tag_block", 1),
        ],
    ),
    (
        "Extract the leading @tag block from a raw IRC wire line",
        [
            ("C:protocol/message.py:Message._parse_tag_block", 2),
            ("C:protocol/message.py:Message.parse", 1),
        ],
    ),
    (
        "Read IRC messages until a stop numeric arrives or a timeout elapses",
        [("C:overview/collector.py:_recv_until", 2)],
    ),
    (
        "Query the IRC server for the NAMES list of a channel with operator flags",
        [("C:overview/collector.py:_query_names", 2)],
    ),
    (
        "Connect to the IRC server and register as an ephemeral observer",
        [
            ("C:overview/collector.py:_connect", 2),
            ("C:overview/collector.py:_handle_registration_line", 1),
        ],
    ),
    (
        "Add agents that are registered in the manifest but not currently on IRC as stopped agents",
        [("C:overview/collector.py:_inject_stopped_agents", 2)],
    ),
    (
        "Read bot configurations from the user's bots directory on disk",
        [("C:overview/collector.py:_collect_bots", 2)],
    ),
    (
        "Create a backend-specific daemon for the codex agent",
        [("C:cli/agents.py:_create_codex_daemon", 2)],
    ),
    (
        "Fail fast with a remediation hint when an agent backend's SDK extra is not installed",
        [("C:cli/agents.py:_require_backend_sdk", 2)],
    ),
    (
        "Archive an agent by stopping it if running and setting an archived flag",
        [("C:cli/agents.py:_cmd_archive", 2)],
    ),
    (
        "Look up an agent by nick and raise an error listing candidates when it is not found",
        [("C:cli/agents.py:_resolve_by_nick", 2)],
    ),
    (
        "Install a systemd or launchd unit that auto-starts the culture console",
        [("C:cli/console.py:_cmd_install", 2)],
    ),
    (
        "Pre-flight check that refuses to start the console when its port is already used by a different target",
        [("C:cli/console.py:_check_port_conflict", 2), ("C:cli/console.py:_same_target", 1)],
    ),
    (
        "Upgrade the culture package using uv or pip and then re-exec the process",
        [("C:cli/mesh.py:_upgrade_culture_package", 2), ("C:cli/mesh.py:_find_upgrade_tool", 1)],
    ),
    (
        "Check that a doctor scan finds on-disk repos with culture.yaml that are not registered in the manifest",
        [
            ("C:doctor/checks.py:check_unregistered", 2),
            ("C:doctor/discovery.py:discover_ondisk_repos", 1),
        ],
    ),
    (
        "Detect installed services that are restart-looping or parked using systemd health properties",
        [("C:doctor/checks.py:check_services", 2), ("C:doctor/checks.py:_service_findings", 1)],
    ),
    (
        "Write a command result to stdout as JSON or text in the CLI output helper",
        [("C:cli/_output.py:emit_result", 2)],
    ),
    (
        "Bypass argparse and forward certain server subcommands to the agentirc CLI",
        [("C:cli/__init__.py:_maybe_forward_to_agentirc", 2)],
    ),
    (
        "Record per-phase timings while awaiting each coroutine of the mesh state collection",
        [("C:overview/collector.py:_timed", 2), ("C:overview/collector.py:collect_mesh_state", 1)],
    ),
]
ISSUE = [
    (
        "ZeroDivisionError: float division by zero in cosine similarity when an embedding comes back as all zeros",
        [("L:assess.py:_cosine_similarity", 2)],
    ),
    (
        "Concurrency ramp never stops and keeps increasing even though requests_per_s has flattened out; the knee is reported at the last step instead of the peak",
        [("L:assess.py:_find_knee", 2), ("L:assess.py:auto_ramp_concurrency", 1)],
    ),
    (
        "Traceback urllib.error.HTTPError: HTTP Error 401: Unauthorized when running assess against the gateway with a blank key; user wants a readable message instead",
        [("L:assess.py:_unauthorized_error", 2), ("L:assess.py:_api_errors", 1)],
    ),
    (
        "Setting GATEWAY_API_KEY to whitespace only still enables auth and every request gets 401",
        [("L:gateway/_config.py:_gateway_api_key", 2)],
    ),
    (
        "LOBES_MESH_SEEDS=http://a:8000/, ,http://b:8000 creates an empty seed and a double slash in the announce URL",
        [("L:gateway/_mesh_config.py:_parse_seeds", 2)],
    ),
    (
        "LOBES_MESH_HEARTBEAT_S=0 crashes the gateway at startup with an unhelpful ValueError; should name the variable and say it must be positive",
        [("L:gateway/_mesh_config.py:_parse_positive_int", 2)],
    ),
    (
        "POST /mesh/announce with a truncated body returns 500: json.decoder.JSONDecodeError: Expecting value: line 1 column 1",
        [
            ("L:gateway/_mesh_routes.py:_parse_json_object", 1),
            ("L:gateway/_mesh_routes.py:MeshRoutes.announce", 2),
        ],
    ),
    (
        "A request with Content-Length: abc hangs the mesh route handler instead of returning an error",
        [("L:gateway/_mesh_routes.py:_read_request_body", 2)],
    ),
    (
        "MeshSchemaIncompatible: schema version 1 expected, got 2.0 when a newer peer announces; members are silently dropped",
        [("L:gateway/_mesh_wire.py:_check_schema_major", 2), ("L:gateway/_mesh_wire.py:decode", 1)],
    ),
    (
        "Two boxes with the same checkpoint but a different speculative_config are treated as one pool; verification should refuse a fingerprint mismatch",
        [
            ("L:gateway/_mesh_routing.py:fingerprints_identical", 2),
            ("L:gateway/_mesh_routing.py:verify_member_roles", 2),
        ],
    ),
    (
        "model=cortex-spark2 returns 404 role_infeasible even though spark2 is verified and in the roster",
        [
            ("L:gateway/_mesh_routing.py:find_member_lane", 2),
            ("L:gateway/_mesh_routing.py:member_lanes", 1),
        ],
    ),
    (
        "After the gateway restarts, a request for a role hosted by a peer answers 404 instead of 503 role_unverified while the peer has not been probed yet",
        [
            ("L:gateway/_mesh_routing.py:_pending_origins_for_role", 2),
            ("L:gateway/_mesh_routing.py:_collect_role_candidates", 1),
        ],
    ),
    (
        "Alias list HAND_ADAPTERS with a trailing comma and a missing equals sign makes the gateway raise instead of ignoring the bad pair",
        [("L:gateway/_config.py:_parse_aliases", 2)],
    ),
    (
        "WARNING: model id claimed by more than one backend, but nothing is printed when the associate and worker share a served name",
        [
            ("L:gateway/_config.py:_warn_on_served_name_collisions", 2),
            ("L:gateway/_config.py:_claimed_model_ids", 2),
        ],
    ),
    (
        "WebSocket connection to /v1/realtime is refused with 400 although the client sent Upgrade: websocket and Connection: keep-alive, upgrade",
        [
            ("L:gateway/_realtime.py:is_websocket_upgrade", 2),
            ("L:gateway/_realtime.py:plan_realtime_upgrade", 1),
        ],
    ),
    (
        "culture channel message 'line1\\nline2' posts a literal backslash-n instead of two lines in the IRC channel",
        [("C:cli/channel.py:_interpret_escapes", 2), ("C:cli/channel.py:_cmd_message", 1)],
    ),
    (
        "culture bot create ../../etc/evil succeeds and writes outside the bots directory",
        [("C:cli/bot.py:_validate_name_or_raise", 2)],
    ),
    (
        "sys.exit('boom') inside an embedded CLI makes culture exit 0 and swallow the message",
        [("C:cli/_passthrough.py:_translate_exit", 2), ("C:cli/_passthrough.py:run", 1)],
    ),
    (
        "culture server start reports started but the port is held by an unrelated process, so agents connect to the wrong server",
        [("C:cli/mesh.py:_wait_for_server_port", 2)],
    ),
    (
        "culture doctor does not warn about a service stuck in a restart loop; systemctl show reports NRestarts=17 and ActiveState=activating",
        [("C:doctor/checks.py:_service_findings", 2), ("C:doctor/checks.py:check_services", 2)],
    ),
    (
        "IRC message with tags @a=b\\:c;d parses to the wrong tag value; escaped semicolon is not decoded",
        [
            ("C:protocol/message.py:_unescape_tag_value", 2),
            ("C:protocol/message.py:Message._parse_tag_block", 2),
        ],
    ),
    (
        "culture overview hangs forever on a dead room: TimeoutError waiting for RPL_ENDOFNAMES never raised",
        [("C:overview/collector.py:_recv_until", 2), ("C:overview/collector.py:_query_names", 1)],
    ),
    (
        "culture console serve fails with 'address already in use' and no hint which culture console owns the port",
        [("C:cli/console.py:_check_port_conflict", 2), ("C:cli/console.py:_port_in_use", 1)],
    ),
    (
        "culture agents start with the codex backend dies with ModuleNotFoundError: No module named 'codex_sdk' and no hint to install the extra",
        [("C:cli/agents.py:_require_backend_sdk", 2), ("C:cli/agents.py:_create_codex_daemon", 1)],
    ),
]


def full(s):
    return (
        s.replace("L:", "lobes-cli:lobes/", 1)
        if s.startswith("L:")
        else s.replace("C:", "culture:culture_core/", 1)
    )


ids = {json.loads(line)["id"] for line in open("corpus.jsonl")}
bad = 0
out = open("queries.jsonl", "w")
for fam, items, pre in (("nl_to_code", NL, "nl"), ("issue_to_source", ISSUE, "is")):
    for i, (q, labs) in enumerate(items, 1):
        rel = {}
        for s, g in labs:
            f = full(s)
            if f not in ids:
                print("MISSING", f)
                bad += 1
            rel[f] = g
        out.write(
            json.dumps(
                {"qid": f"{pre}{i:02d}", "family": fam, "query": q, "relevant": rel},
                ensure_ascii=False,
            )
            + "\n"
        )
print(len(NL), len(ISSUE), bad)
