"""MetaSyn projection selects by full-batch length, before performance is known."""
import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]


class Projection(unittest.TestCase):
    def setUp(self):
        path=ROOT/'benchmarks/eager_parallel_r31/project_meta.py'
        self.assertTrue(path.is_file(),'R31 MetaSyn projector is missing')
        spec=importlib.util.spec_from_file_location('r31_project_test',path)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.lengths=torch.tensor([[8,8,0,0,0,0],[0,0,1,1,1,1],*[ [i]*6 for i in range(2,8)]],dtype=torch.int64)
        self.offsets=torch.cat((torch.zeros(1,dtype=torch.int64),self.lengths.flatten().cumsum(0)))
        self.indices=torch.arange(int(self.offsets[-1]),dtype=torch.int64)+100

    def test_selection_uses_full_batch_mean_midpoint_strata_and_fixed_permutation(self):
        a=self.module.project(self.indices,self.offsets,self.lengths,tables=4,batch=2,vocabulary=17)
        b=self.module.project(self.indices,self.offsets,self.lengths,tables=4,batch=2,vocabulary=17)
        self.assertEqual(a,b)
        self.assertEqual(set(a['selected_source_tables']),{2,3,5,7})
        self.assertEqual(a['selection']['quantile_ranks'],[1,3,5,7])
        self.assertEqual(a['selection']['seed'],20261110)
        self.assertEqual(a['source_shape'],[8,6])

    def test_projection_preserves_ragged_lengths_offsets_duplicate_ids_and_input_bytes(self):
        saved=[x.clone() for x in (self.indices,self.offsets,self.lengths)]
        result=self.module.project(self.indices,self.offsets,self.lengths,tables=8,batch=2,vocabulary=17)
        for row in result['tables']:
            t=row['source_table'];start=int(self.offsets[t*6]);end=int(self.offsets[t*6+2])
            self.assertEqual(row['lengths'],self.lengths[t,:2].tolist())
            self.assertEqual(row['offsets'],(self.offsets[t*6:t*6+2]-start).tolist())
            self.assertEqual(row['indices'],(self.indices[start:end]%17).tolist())
            self.assertEqual(row['mean_length'],(end-start)/2)
        empty=next(row for row in result['tables'] if row['source_table']==1)
        self.assertEqual(empty['indices'],[]);self.assertEqual(empty['offsets'],[0,0])
        for actual,expected in zip((self.indices,self.offsets,self.lengths),saved):
            self.assertTrue(torch.equal(actual,expected))

    def test_malformed_source_offsets_dtype_and_projection_bounds_are_rejected(self):
        changed=self.offsets.clone();changed[1]+=1
        variants=[(self.indices,changed,self.lengths,{}),
                  (self.indices,self.offsets,self.lengths.float(),{}),
                  (self.indices,self.offsets,self.lengths,dict(tables=9)),
                  (self.indices,self.offsets,self.lengths,dict(batch=7)),
                  (self.indices,self.offsets,self.lengths,dict(vocabulary=0))]
        for indices,offsets,lengths,kwargs in variants:
            config=dict(tables=8,batch=2,vocabulary=17);config.update(kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                self.module.project(indices,offsets,lengths,**config)

    def test_official_float32_encoded_ids_are_converted_only_when_lossless(self):
        integer=self.module.project(self.indices,self.offsets,self.lengths,tables=8,batch=2,vocabulary=17)
        encoded=self.module.project(self.indices.float(),self.offsets,self.lengths,tables=8,batch=2,vocabulary=17)
        self.assertEqual(encoded,integer)
        for value in (100.25,float('nan'),float('inf'),float(2**63)):
            changed=self.indices.float();changed[0]=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.module.project(changed,self.offsets,self.lengths,tables=8,batch=2,vocabulary=17)


if __name__=='__main__':
    unittest.main()
