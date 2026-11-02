"""Local behavior checks for load generation and cleanup safety."""

import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controller = load_module("controller", ROOT / "benchmark-aws" / "controller.py")
load = load_module("benchmark_load", ROOT / "benchmark-aws" / "load.py")
RUN_ID = "infer-0123456789abcdef"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps(
            {"request_id": data["request_id"], "output_data": [1.0], "cached": True}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class LoadTests(unittest.TestCase):
    def test_real_local_http_and_summary(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            summary, rows = load.run_level(f"http://127.0.0.1:{server.server_port}", 8, 2, True, 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(summary["requests"], 8)
        self.assertEqual(summary["successes"], 8)
        self.assertEqual(summary["cache_hits"], 8)
        self.assertEqual(summary["failures"], 0)
        self.assertEqual(len({row["request_id"] for row in rows}), 8)
        self.assertIsNotNone(summary["successful_latency_ms"]["p99"])

    def test_connection_failure_is_not_a_success(self):
        with mock.patch.object(load.urllib.request, "urlopen", side_effect=OSError("unavailable")):
            result = load.send_one("http://127.0.0.1:1", 1, False)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "OSError")
        self.assertIsNone(load.percentile([], 0.99))

    def test_invalid_json_shape_is_recorded_as_failure(self):
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = b"[]"
        response.__enter__.return_value = response
        with mock.patch.object(load.urllib.request, "urlopen", return_value=response):
            result = load.send_one("http://127.0.0.1:1", 1, False)
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "invalid_response")

    def test_miss_timings_are_recorded_and_missing_values_stay_unknown(self):
        response = mock.MagicMock()
        response.status = 200
        response.read.return_value = json.dumps(
            {
                "request_id": "bench-1",
                "output_data": [1.0],
                "cached": False,
                "queue_wait_us": 250,
                "batch_execution_us": 4000,
                "batch_size": 2,
            }
        ).encode()
        response.__enter__.return_value = response
        with mock.patch.object(load.urllib.request, "urlopen", return_value=response):
            result = load.send_one("http://127.0.0.1:1", 1, False)
        self.assertTrue(result["success"])
        self.assertEqual(result["queue_wait_us"], 250)
        self.assertEqual(result["batch_execution_us"], 4000)
        self.assertEqual(result["batch_size"], 2)
        response.read.return_value = b'{"request_id":"bench-1","output_data":[1],"cached":true}'
        with mock.patch.object(load.urllib.request, "urlopen", return_value=response):
            cached = load.send_one("http://127.0.0.1:1", 1, True)
        self.assertIsNone(cached["queue_wait_us"])
        self.assertIsNone(cached["batch_execution_us"])

    def test_summary_keeps_missing_timing_samples_separate(self):
        rows = [
            {
                "success": True,
                "latency_ms": 10,
                "cached": False,
                "queue_wait_us": 200,
                "batch_execution_us": 4000,
            },
            {
                "success": True,
                "latency_ms": 2,
                "cached": True,
                "queue_wait_us": None,
                "batch_execution_us": None,
            },
        ]
        with mock.patch.object(load, "send_one", side_effect=rows):
            summary, _ = load.run_level("http://127.0.0.1", 2, 1, False, 0)
        self.assertEqual(summary["worker_timings_us"]["queue_wait_us"]["samples"], 1)
        self.assertEqual(summary["worker_timings_us"]["queue_wait_us"]["p50"], 200)
        self.assertEqual(summary["worker_timings_us"]["batch_execution_us"]["p50"], 4000)


class SafetyTests(unittest.TestCase):
    def test_remote_failure_is_reported_before_missing_result_files(self):
        def fake_command(args, **kwargs):
            return subprocess.CompletedProcess(args, 1 if args[0] == "ssh" else 0)

        with tempfile.TemporaryDirectory() as temp:
            with (
                mock.patch.object(controller, "command", side_effect=fake_command),
                mock.patch.object(controller, "verify_archive") as verify,
            ):
                with self.assertRaisesRegex(RuntimeError, "remote workload returned 1"):
                    controller.execute_and_collect("example", [], Path(temp))
                verify.assert_not_called()

    def test_cleanup_refuses_wrong_account_before_resource_calls(self):
        state = {"region": controller.REGION, "account": "expected", "run_id": RUN_ID}
        with (
            mock.patch.object(controller, "identity", return_value="different"),
            mock.patch.object(controller, "tagged_instances") as instances,
        ):
            with self.assertRaises(RuntimeError):
                controller.cleanup(state)
            instances.assert_not_called()

    def test_cleanup_targets_only_discovered_tagged_resources(self):
        state = {"region": controller.REGION, "account": "123", "run_id": RUN_ID}
        with (
            mock.patch.object(controller, "identity", return_value="123"),
            mock.patch.object(
                controller,
                "tagged_instances",
                side_effect=[
                    [{"InstanceId": "i-owned", "State": {"Name": "running"}}],
                    [{"InstanceId": "i-owned", "State": {"Name": "terminated"}}],
                ],
            ),
            mock.patch.object(
                controller, "tagged_groups", side_effect=[[{"GroupId": "sg-owned"}], []]
            ),
            mock.patch.object(
                controller,
                "tagged_volumes",
                side_effect=[[{"VolumeId": "vol-owned", "State": "available"}], []],
            ),
            mock.patch.object(controller, "wait_for_termination", return_value=True),
            mock.patch.object(controller, "aws") as aws,
        ):
            self.assertTrue(controller.cleanup(state))
        aws.assert_any_call("ec2", "terminate-instances", "--instance-ids", "i-owned")
        aws.assert_any_call("ec2", "delete-volume", "--volume-id", "vol-owned")
        aws.assert_any_call("ec2", "delete-security-group", "--group-id", "sg-owned")

    def test_missing_instance_is_not_confirmation_of_termination(self):
        state = {
            "region": controller.REGION,
            "account": "123",
            "run_id": RUN_ID,
            "instance_id": "i-owned",
        }
        running = [{"InstanceId": "i-owned", "State": {"Name": "running"}}]
        with (
            mock.patch.object(controller, "identity", return_value="123"),
            mock.patch.object(controller, "tagged_instances", return_value=running),
            mock.patch.object(controller, "aws", return_value={"Reservations": []}),
            mock.patch.object(controller.time, "monotonic", side_effect=[0, 241]),
        ):
            self.assertFalse(controller.cleanup(state))

    def test_wait_checks_recorded_instance_until_terminated(self):
        def response(status):
            return {
                "Reservations": [
                    {"Instances": [{"InstanceId": "i-owned", "State": {"Name": status}}]}
                ]
            }

        with (
            mock.patch.object(
                controller, "aws", side_effect=[response("shutting-down"), response("terminated")]
            ) as aws,
            mock.patch.object(controller.time, "sleep"),
        ):
            self.assertTrue(controller.wait_for_termination(["i-owned"]))
        self.assertEqual(aws.call_count, 2)

    def test_wait_does_not_treat_not_found_as_terminated(self):
        error = subprocess.CalledProcessError(1, "aws", stderr="InvalidInstanceID.NotFound")
        with (
            mock.patch.object(controller, "aws", side_effect=error),
            mock.patch.object(controller.time, "monotonic", side_effect=[0, 241]),
        ):
            self.assertFalse(controller.wait_for_termination(["i-owned"]))

    def test_recorded_instance_lookup_waits_for_ownership_tag(self):
        instance = {"InstanceId": "i-owned", "State": {"Name": "running"}}
        tagged = {**instance, "Tags": [{"Key": controller.TAG_KEY, "Value": RUN_ID}]}
        responses = [
            {"Reservations": [{"Instances": [instance]}]},
            {"Reservations": [{"Instances": [tagged]}]},
        ]
        with (
            mock.patch.object(controller, "aws", side_effect=responses),
            mock.patch.object(controller.time, "sleep"),
        ):
            self.assertEqual(controller.find_recorded_instance("i-owned", RUN_ID), tagged)

    def test_recorded_instance_with_wrong_tag_is_never_accepted(self):
        response = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-owned",
                            "Tags": [{"Key": controller.TAG_KEY, "Value": "other"}],
                        }
                    ]
                }
            ]
        }
        with (
            mock.patch.object(controller, "aws", return_value=response),
            mock.patch.object(controller.time, "monotonic", side_effect=[0, 31]),
        ):
            with self.assertRaisesRegex(RuntimeError, "ownership"):
                controller.find_recorded_instance("i-owned", RUN_ID)

    def test_unknown_launch_outcome_is_not_reported_as_clean(self):
        state = {
            "region": controller.REGION,
            "account": "123",
            "run_id": RUN_ID,
            "launch_attempted": True,
        }
        with (
            mock.patch.object(controller, "identity", return_value="123"),
            mock.patch.object(controller, "tagged_instances", return_value=[]),
            mock.patch.object(controller, "tagged_groups", return_value=[]),
            mock.patch.object(controller, "tagged_volumes", return_value=[]),
        ):
            self.assertFalse(controller.cleanup(state))

    def test_cleanup_deletes_group_that_appears_after_a_delayed_listing(self):
        state = {
            "region": controller.REGION,
            "account": "123",
            "run_id": RUN_ID,
            "group_id": "sg-owned",
        }
        with (
            mock.patch.object(controller, "identity", return_value="123"),
            mock.patch.object(controller, "tagged_instances", return_value=[]),
            mock.patch.object(controller, "tagged_volumes", return_value=[]),
            mock.patch.object(
                controller, "tagged_groups", side_effect=[[], [{"GroupId": "sg-owned"}], []]
            ),
            mock.patch.object(controller, "resource_absent", side_effect=[False, False, True]),
            mock.patch.object(controller.time, "sleep"),
            mock.patch.object(controller, "aws") as aws,
        ):
            self.assertTrue(controller.cleanup(state))
        aws.assert_called_once_with("ec2", "delete-security-group", "--group-id", "sg-owned")

    def test_userdata_rejects_shell_metacharacters_in_public_key(self):
        with self.assertRaises(ValueError):
            controller.user_data("ssh-ed25519 invalid'payload")
        result = controller.user_data("ssh-ed25519 YWJj comment")
        self.assertIn("shutdown -h +60", result)
        self.assertNotIn(" comment'", result)

    def test_archive_checksum_and_path_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, digest = root / "result.tar.gz", root / "digest"
            with tarfile.open(archive, "w:gz") as tar:
                for name, contents in [
                    ("results/summary.json", b"[]"),
                    ("results/requests.jsonl", b"{}\n"),
                    ("results/exit-status", b"0\n"),
                ]:
                    info = tarfile.TarInfo(name)
                    info.size = len(contents)
                    tar.addfile(info, io.BytesIO(contents))
            digest.write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "  file\n")
            controller.verify_archive(archive, digest, root / "out")
            self.assertTrue((root / "out/results/summary.json").exists())
            digest.write_text("0" * 64)
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                controller.verify_archive(archive, digest, root / "bad")

    def test_archive_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive, digest = root / "result.tar.gz", root / "digest"
            with tarfile.open(archive, "w:gz") as tar:
                for name in (
                    "results/summary.json",
                    "results/requests.jsonl",
                    "results/exit-status",
                    "../escape",
                ):
                    info = tarfile.TarInfo(name)
                    info.size = 1
                    tar.addfile(info, io.BytesIO(b"0"))
            digest.write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
            with self.assertRaisesRegex(RuntimeError, "unsafe archive"):
                controller.verify_archive(archive, digest, root / "out")
            self.assertFalse((root / "escape").exists())

    def test_run_failure_attempts_cleanup_and_removes_ephemeral_key(self):
        context = {
            "region": controller.REGION,
            "account": "123",
            "public_ip": "127.0.0.1",
            "vpc": "vpc-1",
            "subnet": "subnet-1",
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def fake_command(args, **_kwargs):
                key = Path(args[args.index("-f") + 1])
                key.write_text("private key")
                key.with_suffix(".pub").write_text("ssh-ed25519 YWJj benchmark")
                return subprocess.CompletedProcess(args, 0)

            def fake_aws(*args, **_kwargs):
                if args[:2] == ("ec2", "create-security-group"):
                    return {"GroupId": "sg-owned"}
                raise subprocess.CalledProcessError(1, args, stderr="denied")

            with (
                mock.patch.object(controller, "HERE", root),
                mock.patch.object(controller, "preflight", return_value=context),
                mock.patch.object(
                    controller, "prepare_source", return_value={"patch_sha256": "test"}
                ),
                mock.patch.object(controller.secrets, "token_hex", return_value="0123456789abcdef"),
                mock.patch.object(controller, "command", side_effect=fake_command),
                mock.patch.object(controller, "aws", side_effect=fake_aws),
                mock.patch.object(controller, "cleanup", return_value=True) as cleanup,
            ):
                self.assertEqual(controller.run(), 1)
            directory = root / "runs" / RUN_ID
            cleanup.assert_called_once()
            self.assertFalse((directory / "id_ed25519").exists())
            state = json.loads((directory / "state.json").read_text())
            self.assertTrue(state["cleanup_verified"])
            self.assertFalse(state["benchmark_succeeded"])

    def test_launch_is_single_instance_and_failure_still_cleans(self):
        context = {
            "region": controller.REGION,
            "account": "123",
            "public_ip": "127.0.0.1",
            "vpc": "vpc-1",
            "subnet": "subnet-1",
        }
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def fake_command(args, **_kwargs):
                key = Path(args[args.index("-f") + 1])
                key.write_text("private key")
                key.with_suffix(".pub").write_text("ssh-ed25519 YWJj benchmark")
                return subprocess.CompletedProcess(args, 0)

            def fake_aws(*args, **_kwargs):
                calls.append(args)
                if args[:2] == ("ec2", "create-security-group"):
                    return {"GroupId": "sg-owned"}
                if args[:2] == ("ec2", "run-instances"):
                    raise subprocess.CalledProcessError(1, args, stderr="launch denied")
                return {}

            with (
                mock.patch.object(controller, "HERE", Path(temp)),
                mock.patch.object(controller, "preflight", return_value=context),
                mock.patch.object(
                    controller, "prepare_source", return_value={"patch_sha256": "test"}
                ),
                mock.patch.object(controller.secrets, "token_hex", return_value="0123456789abcdef"),
                mock.patch.object(controller, "command", side_effect=fake_command),
                mock.patch.object(controller, "aws", side_effect=fake_aws),
                mock.patch.object(controller, "cleanup", return_value=True) as cleanup,
            ):
                self.assertEqual(controller.run(), 1)
            launch = next(call for call in calls if call[:2] == ("ec2", "run-instances"))
            self.assertEqual(launch[launch.index("--count") + 1], "1")
            self.assertNotIn("--min-count", launch)
            cleanup.assert_called_once()

    def test_cleanup_rejects_non_run_tag(self):
        for value in ("infer-../../etc", "", "infer-0123456789abcdeg"):
            with self.assertRaises(ValueError):
                controller.filters(value)


if __name__ == "__main__":
    unittest.main()
