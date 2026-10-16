import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def run_launcher(self, name="train.sh", extra_env=None, args=()):
        env = {key: value for key, value in os.environ.items()
               if key in {"PATH", "TMPDIR", "HOME", "LANG", "SYSTEMROOT"}}
        env.update({"DATASET_TYPE": "webqsp", "MODEL_PATH": "example/sft-model",
                    "TRAIN_FILE": "/path with spaces/train.parquet", "VAL_FILE": "/path with spaces/dev.parquet",
                    "NGPUS_PER_NODE": "8", "GAPD_DRY_RUN": "true", "TIMESTAMP": "unit-check"})
        env.update(extra_env or {})
        return subprocess.run(["bash", str(ROOT / "scripts" / name), *args],
                              cwd="/tmp", env=env, text=True, capture_output=True, timeout=30)

    def test_portable_paths_and_main_recipe(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("data.train_files=['/path with spaces/train.parquet']", result.stdout)
        self.assertIn("+actor_rollout_ref.actor.gapd_jsd_coef=0.5", result.stdout)
        self.assertIn("+actor_rollout_ref.rollout.gapd_skip_same_next_state=true", result.stdout)
        self.assertIn("+actor_rollout_ref.rollout.gapd_skip_expression_id_risk=true", result.stdout)
        self.assertIn(f"custom_reward_function.path={ROOT}/kbqa_custom_reward.py", result.stdout)

    def test_missing_explicit_validation_is_rejected(self):
        result = self.run_launcher(extra_env={"VAL_FILE": ""})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VAL_FILE", result.stderr)

    def test_evaluation_disables_distillation(self):
        result = self.run_launcher("evaluate.sh", {"GAPD_ENABLE": "true", "OPD_ENABLE": "true"})
        self.assertEqual(result.returncode, 0, result.stderr)
        for argument in ("trainer.val_only=true", "trainer.resume_mode=disable",
                         "+actor_rollout_ref.actor.gapd_enable=false",
                         "+actor_rollout_ref.actor.opd_enable=false",
                         "+actor_rollout_ref.rollout.gapd_val_teacher_debug=false"):
            self.assertIn(argument, result.stdout)

    def test_override_and_password_redaction(self):
        password = "-".join(("local", "dry", "run", "credential"))
        result = self.run_launcher(extra_env={"FREEBASE_ODBC_PWD": password, "TOTAL_EPOCHS": "8"},
                                   args=("trainer.total_epochs=1",))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(password, result.stdout + result.stderr)
        self.assertIn("+odbc_config.pwd=<redacted>", result.stdout)
        self.assertIn("trainer.total_epochs=8", result.stdout)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "trainer.total_epochs=1")


if __name__ == "__main__":
    unittest.main()
