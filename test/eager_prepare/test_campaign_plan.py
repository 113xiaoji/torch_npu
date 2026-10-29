import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE=Path(__file__).parents[2]/"benchmarks/eager_prepare/campaign_plan.py"


class PlanTests(unittest.TestCase):
    def test_plan_is_fixed48_fresh_and_reverses_pair_order_without_writing(self):
        self.assertTrue(SOURCE.is_file(),"R24 campaign planner not implemented")
        spec=importlib.util.spec_from_file_location("r24_plan_test",SOURCE)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for abi in ("std","ft"):
                path=root/(abi+".json");path.write_text(json.dumps(dict(schema=1,abi=abi,python="/venv/"+abi+"/bin/python")));paths.append(path)
            args=module.parse_args(["--std-artifact",str(paths[0]),"--ft-artifact",str(paths[1]),"--output-root",str(root/"new")])
            plan=module.make_plan(args)
            self.assertEqual(len(plan["cases"]),48)
            self.assertEqual(len({r["output"] for r in plan["cases"]}),48)
            self.assertTrue(all(r["fresh_process"] and r["samples"]==2000 and r["warmup"]==20 for r in plan["cases"]))
            group=[r for r in plan["cases"] if r["abi"]=="std" and r["model"]=="recommendation"]
            self.assertEqual([r["mode"]for r in group],["off","on","on","off","off","on"])
            self.assertFalse((root/"new").exists())


if __name__=="__main__":unittest.main()
