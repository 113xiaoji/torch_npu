"""A predeclared length-stratified projection of the public MetaSyn trace.

Data: facebookresearch/dlrm_datasets (MIT), 2021 base trace at SOURCE_GIT.
The original 26-table model is retained by mapping source IDs modulo 1024.
This projection does not reproduce Meta's model size or private production data.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

SOURCE_GIT = '6f46ba7906e47a2f6bab6499056480d07d784eb8'
SOURCE_SHA256 = '1d4d3ca295fade795cbf973d9b29e721b7c76a2130684582d9f4ab579c95e772'
SOURCE_BYTES = 2398183998
SEED = 20261110


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def project(indices, offsets, lengths, *, tables=26, batch=128, vocabulary=1024, seed=SEED):
    import torch
    for name,value in (('tables',tables),('batch',batch),('vocabulary',vocabulary),('seed',seed)):
        require(type(value) is int and value >= (0 if name=='seed' else 1), 'invalid '+name)
    for tensor,rank,encoded_ids in ((indices,1,True),(offsets,1,False),(lengths,2,False)):
        allowed=(torch.int32,torch.int64,torch.float32) if encoded_ids else (torch.int32,torch.int64)
        require(tensor.device.type=='cpu' and tensor.dtype in allowed
                and tensor.ndim==rank, 'CPU source tensors with original types/ranks required')
    source_tables,source_batch=lengths.shape
    require(tables<=source_tables and batch<=source_batch, 'projection exceeds source shape')
    require(offsets.numel()==lengths.numel()+1 and int(offsets[0])==0
            and int(offsets[-1])==indices.numel(), 'source offsets do not span indices')
    require(bool((lengths>=0).all()) and torch.equal(torch.diff(offsets),lengths.flatten()),
            'source offsets disagree with nonnegative lengths')
    totals=lengths.to(torch.int64).sum(dim=1).tolist()
    ordered=sorted(range(source_tables),key=lambda i:(totals[i],i))
    ranks=[((2*j+1)*source_tables)//(2*tables) for j in range(tables)]
    selected=[ordered[rank] for rank in ranks]
    random.Random(seed).shuffle(selected)
    rows=[]
    for model_table,source_table in enumerate(selected):
        at=source_table*source_batch;start=int(offsets[at]);end=int(offsets[at+batch])
        original_values=indices[start:end]
        values=original_values.to(torch.int64)
        # The official base trace stores IDs as float32. Reject fractional,
        # nonfinite or overflowing values rather than silently truncating them.
        require(torch.equal(values.to(original_values.dtype),original_values),
                'source IDs cannot be losslessly represented as int64')
        require(not values.numel() or int(values.min())>=0, 'negative source ID')
        row=dict(model_table=model_table,source_table=source_table,
            full_mean_length=totals[source_table]/source_batch,
            mean_length=(end-start)/batch, indices_total=end-start,
            lengths=lengths[source_table,:batch].tolist(),
            offsets=(offsets[at:at+batch]-start).to(torch.int64).tolist(),
            indices=(values%vocabulary).tolist())
        rows.append(row)
    return dict(schema=1,source_git=SOURCE_GIT,source_sha256=SOURCE_SHA256,
        source_shape=[source_tables,source_batch],batch=batch,vocabulary=vocabulary,
        selection=dict(policy='full-batch mean midpoint quantiles then fixed permutation',
            quantile_ranks=ranks,seed=seed,source_bag_start=0,source_bag_count=batch),
        selected_source_tables=selected,tables=rows,
        source_stats=dict(mean_length=sum(totals)/(source_tables*source_batch),
            min_table_mean=min(totals)/source_batch,max_table_mean=max(totals)/source_batch))


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--source-gzip',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    require(args.source_gzip.stat().st_size==SOURCE_BYTES and sha(args.source_gzip)==SOURCE_SHA256,
            'official source gzip identity differs')
    require(not args.output.exists(), 'new projection output required')
    unpacked=args.source_gzip.with_suffix('')
    receipt=unpacked.with_suffix('.decompressed.json')
    if unpacked.exists():
        previous=json.loads(receipt.read_text())
        require(previous['source_sha256']==SOURCE_SHA256 and previous['sha256']==sha(unpacked),
                'existing decompression identity differs')
    else:
        digest=hashlib.sha256();count=0;started=time.monotonic();last=0
        with gzip.open(args.source_gzip,'rb') as source,unpacked.open('xb') as target:
            while block:=source.read(8*1024*1024):
                target.write(block);digest.update(block);count+=len(block)
                if count-last>=1024**3:
                    print(json.dumps(dict(decompressed_bytes=count,seconds=time.monotonic()-started)),flush=True)
                    last=count
        receipt.write_text(json.dumps(dict(source_sha256=SOURCE_SHA256,sha256=digest.hexdigest(),
            bytes=count),indent=2)+'\n')
    import torch
    torch.set_num_threads(8)
    try:
        indices,offsets,lengths=torch.load(unpacked,map_location='cpu',weights_only=True,mmap=True)
    except RuntimeError as error:
        if 'mmap' not in str(error):
            raise
        indices,offsets,lengths=torch.load(unpacked,map_location='cpu',weights_only=True)
    require(tuple(lengths.shape)==(856,65536), 'official trace shape differs')
    result=project(indices,offsets,lengths)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,sort_keys=True,separators=(',',':'));stream.write('\n')
    summary={key:value for key,value in result.items() if key!='tables'}
    summary['tables']=[{key:value for key,value in row.items() if key not in ('indices','offsets','lengths')}
                       for row in result['tables']]
    summary.update(passed=True,projection_sha256=sha(args.output),projection_bytes=args.output.stat().st_size,
        selected_mean_length=sum(row['indices_total'] for row in result['tables'])/(26*128),
        threshold64_side_tables=[row['model_table'] for row in result['tables'] if row['indices_total']//128>=64])
    args.output.with_name('projection-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
