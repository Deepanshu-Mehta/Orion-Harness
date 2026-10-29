"""Day 2 — tests for the operations layer.

Concept: the jail, process hygiene and background jobs are proved on a real
temp workspace; the sandbox is proved by trying to escape it.

Design rules: LocalEnv tests always run; sandbox tests skip when the backend
is missing; docker needs ORION_TEST_DOCKER=1 and ssh needs
ORION_TEST_SSH=host:/abs/path. Skips name their reason.
"""

import os
import shutil
import signal
import socket
import tempfile
import time
import unittest
from unittest import mock

from orion.env import DockerEnv, LocalEnv, SandboxEnv, SSHEnv, make_env


class EnvContract:
    """Checks every environment must pass; mixed into per-backend cases."""

    def make(self, root):
        raise NotImplementedError

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="orion-env-")
        self.env = self.make(self.tmp)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.addCleanup(self.env.close)

    def test_resolve_escapes_raise(self):
        for bad in ("../x", "/etc/passwd", "a/../../x"):
            with self.subTest(path=bad), self.assertRaises(PermissionError):
                self.env.resolve(bad)

    def test_resolve_nested_new_path(self):
        self.assertEqual(self.env.resolve("a/b/new.txt"), self.env.root + "/a/b/new.txt")

    def test_file_roundtrip(self):
        self.env.write_bytes("d/e/f.txt", b"hello\n")
        self.assertTrue(self.env.exists("d/e/f.txt"))
        self.assertTrue(self.env.is_dir("d/e"))
        self.assertEqual(self.env.read_bytes("d/e/f.txt"), b"hello\n")
        self.env.write_bytes("node_modules/x.js", b"hidden")
        self.assertEqual(self.env.list_files(100), ["d/e/f.txt"])
        self.assertEqual(self.env.search("hel+o", "**/*.txt", 10), [("d/e/f.txt", 1, "hello")])

    def test_exec_basics(self):
        r = self.env.exec("echo out; echo err >&2; exit 3", timeout=30)
        self.assertEqual((r.exit_code, r.timed_out), (3, False))
        self.assertIn("out", r.output)
        self.assertIn("err", r.output)

    def test_stdin_is_closed(self):
        start = time.monotonic()
        r = self.env.exec("read x; echo got:$x", timeout=10)
        self.assertLess(time.monotonic() - start, 5)
        self.assertIn("got:", r.output)

    def test_timeout_returns_partial_output(self):
        start = time.monotonic()
        r = self.env.exec("echo before; sleep 30", timeout=1)
        self.assertLess(time.monotonic() - start, 4)
        self.assertTrue(r.timed_out)
        self.assertIsNone(r.exit_code)
        self.assertIn("before", r.output)

    def test_output_cap(self):
        r = self.env.exec("python3 -c \"print('x' * 100000)\"", timeout=30)
        self.assertTrue(r.truncated)
        self.assertLess(len(r.output), 30_200)
        self.assertIn("chars truncated", r.output)


class LocalEnvTests(EnvContract, unittest.TestCase):
    def make(self, root):
        return LocalEnv(root)

    def test_symlink_out_of_root_raises(self):
        outside = tempfile.mkdtemp(prefix="orion-outside-")
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        os.symlink(outside, os.path.join(self.tmp, "link"))
        with self.assertRaises(PermissionError):
            self.env.resolve("link/secret.txt")
        with self.assertRaises(PermissionError):
            self.env.read_bytes("link/secret.txt")

    def test_absolute_inside_root_ok(self):
        inside = os.path.join(self.env.root, "x.txt")
        self.assertEqual(self.env.resolve(inside), inside)

    def test_secrets_scrubbed_from_exec(self):
        with mock.patch.dict(os.environ, {"ORION_API_KEY": "sekrit", "MY_TOKEN": "t",
                                          "DB_PASSWORD": "p"}):
            r = self.env.exec("echo [$ORION_API_KEY][$MY_TOKEN][$DB_PASSWORD]; env", timeout=10)
        self.assertIn("[][][]", r.output)
        self.assertNotIn("sekrit", r.output)
        self.assertIn("CI=1", r.output)

    def test_background_grandchild_does_not_block(self):
        start = time.monotonic()
        r = self.env.exec("(sleep 30 &) ; echo hi", timeout=20)
        self.assertLess(time.monotonic() - start, 5)
        self.assertIn("hi", r.output)

    def test_spawn_output_and_stop(self):
        job = self.env.spawn("python3 -m http.server 0 --bind 127.0.0.1")
        self.assertEqual(job, "job_1")
        output, running = "", False
        for _ in range(50):
            output, running = self.env.job_output(job, 2000)
            if "Serving HTTP" in output:
                break
            time.sleep(0.1)
        self.assertTrue(running)
        self.assertIn("Serving HTTP", output)
        self.assertEqual(self.env.list_jobs()[0][0], job)
        pgid = self.env._jobs.get(job)["proc"].pid
        self.assertIn("stopped", self.env.job_stop(job))
        self.assertFalse(self.env.job_output(job, 100)[1])
        with self.assertRaises(ProcessLookupError):
            for _ in range(20):
                os.killpg(pgid, 0)
                time.sleep(0.1)

    def test_close_stops_jobs(self):
        job = self.env.spawn("sleep 30")
        self.env.close()
        self.assertFalse(self.env.job_output(job, 100)[1])


@unittest.skipUnless(SandboxEnv.available(), "sandbox backend (sandbox-exec / bwrap) not installed")
class SandboxEnvTests(EnvContract, unittest.TestCase):
    def make(self, root):
        return SandboxEnv(root)

    def test_write_inside_root_ok(self):
        r = self.env.exec("echo ok > inside.txt && cat inside.txt", timeout=10)
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(self.env.read_bytes("inside.txt"), b"ok\n")

    def test_write_home_denied(self):
        target = os.path.expanduser("~/orion-escape-test")
        self.addCleanup(lambda: os.path.exists(target) and os.unlink(target))
        r = self.env.exec(f"echo x > {target}", timeout=10)
        self.assertNotEqual(r.exit_code, 0)
        self.assertFalse(os.path.exists(target))

    def test_ssh_dir_unreadable(self):
        r = self.env.exec("ls ~/.ssh/ && cat ~/.ssh/* ~/.ssh/config", timeout=10)
        self.assertNotEqual(r.exit_code, 0)

    def test_env_file_unreadable(self):
        self.env.write_bytes(".env", b"SECRET=1\n")
        r = self.env.exec("cat .env", timeout=10)
        self.assertNotEqual(r.exit_code, 0)
        self.assertNotIn("SECRET", r.output)

    def test_orion_dir_read_only(self):
        r = self.env.exec("echo x > .orion/x", timeout=10)
        self.assertNotEqual(r.exit_code, 0)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, ".orion", "x")))

    def test_network_denied(self):
        r = self.env.exec(
            "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 443), 3)\"",
            timeout=15)
        self.assertNotEqual(r.exit_code, 0)

    def test_read_only_denies_every_write(self):
        ro = SandboxEnv(self.tmp, writable=False)
        self.addCleanup(ro.close)
        r = ro.exec("echo x > a.txt", timeout=10)
        self.assertNotEqual(r.exit_code, 0)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "a.txt")))
        self.assertEqual(ro.exec("echo tmp-ok > /tmp/orion-ro-test && echo yes", 10).exit_code, 0)

    def test_spawn_in_sandbox(self):
        job = self.env.spawn("echo started; sleep 30")
        for _ in range(30):
            output, running = self.env.job_output(job, 200)
            if "started" in output:
                break
            time.sleep(0.1)
        self.assertIn("started", output)
        self.env.job_stop(job)
        self.assertFalse(self.env.job_output(job, 100)[1])


@unittest.skipUnless(os.environ.get("ORION_TEST_DOCKER") == "1", "set ORION_TEST_DOCKER=1 to run")
class DockerEnvTests(EnvContract, unittest.TestCase):
    def make(self, root):
        return DockerEnv(root)

    def test_root_is_container_path(self):
        self.assertEqual(self.env.root, "/workspace")
        self.assertIn("/workspace", self.env.exec("pwd", 30).output)

    def test_network_off(self):
        r = self.env.exec(
            "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 443), 3)\"", 30)
        self.assertNotEqual(r.exit_code, 0)


def _ssh_target():
    spec = os.environ.get("ORION_TEST_SSH", "")
    host, _, root = spec.partition(":")
    return host, root


@unittest.skipUnless(all(_ssh_target()), "set ORION_TEST_SSH=host:/abs/path to run")
class SSHEnvTests(EnvContract, unittest.TestCase):
    def setUp(self):
        host, root = _ssh_target()
        self.tmp = root.rstrip("/") + f"/orion-test-{os.getpid()}"
        self.env = SSHEnv(host, self.tmp)
        self.env._check(f"mkdir -p {self.tmp}")
        self.addCleanup(self.env.close)
        self.addCleanup(self.env._ssh, f"rm -rf {self.tmp}")


class FactoryTests(unittest.TestCase):
    def test_specs(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        self.assertIsInstance(make_env("local", tmp), LocalEnv)
        ssh = make_env("ssh:user@box:/srv/work", tmp, disposable=True)
        self.assertEqual((ssh.host, ssh.root, ssh.disposable), ("user@box", "/srv/work", True))
        with self.assertRaises(ValueError):
            make_env("ftp", tmp)


if __name__ == "__main__":
    unittest.main()
